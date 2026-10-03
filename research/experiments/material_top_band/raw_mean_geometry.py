"""Explicit manufactured half-prism raw means and their moving mass metric."""
import math

import numpy as np

from . import inventory_pressure as inventory
from .real_geometry import _scalar
from .slope_dual_stock import readonly

AUTHORITY = 'manufactured_half_prism_raw_means'


class RawMeanGeometry:
    def __init__(self, profile, *, authority, distance_m=2., length_m=1.5):
        if authority != AUTHORITY:
            raise ValueError('explicit manufactured raw mean authority required')
        self.authority = authority
        self.distance = _scalar(distance_m, 'distance', positive=True)
        self.length = _scalar(length_m, 'length', positive=True)
        self.profile = inventory.reconstruct(profile.state, eos=profile.eos, representation=profile.representation,
                                             external_pressure_Pa=profile.external_pressure_Pa)
        state = self.profile.state
        if state.h.shape != (2, 14) or not np.all(state.wet_mask) or state.eta[0] != state.eta[1] or state.bottom[0] != state.bottom[1]:
            raise ValueError('two all-wet fourteen-layer flat half-prisms required')
        self.area = self.distance * self.length / 2.
        if not math.isfinite(self.area) or self.area <= 0.:
            raise ValueError('finite positive physical area required')
        z, h = state.interfaces, state.h
        overlap = np.maximum(0., np.minimum(z[0, :-1, None], z[1, None, :-1]) - np.maximum(z[0, 1:, None], z[1, None, 1:]))
        K = np.zeros((28, 28))
        K[:14, 14:], K[14:, :14] = overlap / h[0, :, None], overlap.T / h[1, :, None]
        self.overlap, self.K = readonly(overlap), readonly(K)
        self.D = readonly(np.diag(profile.eos.rho0 * self.area * h.ravel()))
        if not np.isfinite(self.D).all() or np.any(self.D.diagonal() <= 0.):
            raise ValueError('finite positive physical mass required')
        self.Q = readonly(.75 * np.eye(28) + .25 * K)
        self.inverse = readonly(np.linalg.inv(self.Q))
        self.endpoint_metric = readonly(self.D @ (2. * np.eye(28) + K) / 3.)
        self.M = readonly(self.inverse.T @ self.endpoint_metric @ self.inverse)
        self.R = readonly(self.M / self.D.diagonal()[None, :])
        self.means = readonly((state.stocks[..., 2:] / (profile.eos.rho0 * h[..., None])).reshape(28, 2))
        root = np.diag(1. / np.sqrt(self.D.diagonal()))
        eigenvalues = np.linalg.eigvalsh(root @ self.M @ root)
        if eigenvalues.min() < 1. - 512. * np.finfo(float).eps or eigenvalues.max() > 4. / 3. + 512. * np.finfo(float).eps:
            raise ValueError('raw mean metric violates positive covariance bounds')

    def direction(self, eta_dot):
        speed = _scalar(eta_dot, 'eta_dot')
        h = self.profile.state.h
        hd = np.zeros((2, 14))
        hd[:, 0] = speed
        od = np.zeros((14, 14))
        od[0, 0] = speed
        Qd = np.zeros((28, 28))
        Qd[:14, 14:] = .25 * (od / h[0, :, None] - self.overlap * hd[0, :, None] / h[0, :, None]**2)
        Qd[14:, :14] = .25 * (od.T / h[1, :, None] - self.overlap.T * hd[1, :, None] / h[1, :, None]**2)
        J, Jd = self.inverse, -self.inverse @ Qd @ self.inverse
        Wd = np.zeros((28, 28))
        Wd[np.ix_([0, 14], [0, 14])] = self.profile.eos.rho0 * self.distance * self.length * speed * np.array([[1./3., 1./6.], [1./6., 1./3.]])
        Md = Jd.T @ self.endpoint_metric @ J + J.T @ Wd @ J + J.T @ self.endpoint_metric @ Jd
        Dd = np.diag(self.profile.eos.rho0 * self.area * hd.ravel())
        Rd = Md / self.D.diagonal()[None, :] - self.R @ (Dd / self.D.diagonal()[None, :])
        return {name: readonly(value) for name, value in dict(Q=Qd, inverse=Jd, endpoint_metric=Wd, M=Md, D=Dd, R=Rd).items()}

    def fd_envelope(self, name, radius):
        """Central difference bound from interval third derivatives, before probing."""
        radius = _scalar(radius, 'FD radius', positive=True)
        h, overlap_matrix = self.profile.state.h, self.overlap
        hd = np.zeros_like(h)
        hd[:, 0] = 1.
        od = np.zeros_like(overlap_matrix)
        od[0, 0] = 1.
        minimum = h - radius * hd
        if np.any(minimum <= 0):
            raise ValueError('FD interval leaves positive geometry')
        q = [0.]
        for order in range(1, 4):
            blocks = []
            for side, overlap in enumerate((overlap_matrix, overlap_matrix.T)):
                constant = abs(od.T if side else od) * h[side, :, None] - overlap * hd[side, :, None]
                blocks.append(.25 * math.factorial(order) * hd[side, :, None]**(order - 1) * abs(constant) / minimum[side, :, None]**(order + 1))
            q.append(math.sqrt(sum(np.sum(block**2) for block in blocks)))
        j0 = np.linalg.norm(self.inverse, 2)
        if j0 * q[1] * radius >= 1.:
            raise ValueError('FD inverse interval is unbounded')
        j0 /= 1. - j0 * q[1] * radius
        j1 = j0**2 * q[1]
        j2 = 2. * j0**3 * q[1]**2 + j0**2 * q[2]
        j3 = 6. * j0**4 * q[1]**3 + 6. * j0**3 * q[1] * q[2] + j0**2 * q[3]
        w1 = self.profile.eos.rho0 * self.area
        w0 = np.linalg.norm(self.endpoint_metric, 2) + radius * w1
        m0 = j0**2 * w0
        m1 = 2. * j1 * w0 * j0 + j0**2 * w1
        m2 = 2. * j2 * w0 * j0 + 2. * j1**2 * w0 + 4. * j1 * w1 * j0
        m3 = 2. * j3 * w0 * j0 + 6. * j2 * w0 * j1 + 6. * j2 * w1 * j0 + 6. * j1**2 * w1
        mass = self.profile.eos.rho0 * self.area
        e = [max(1. / (mass * minimum.ravel()))]
        e.extend(max(math.factorial(n) * hd.ravel() / (mass * minimum.ravel()**(n+1))) for n in range(1, 4))
        third = dict(Q=q[3], M=m3, R=m3 * e[0] + 3. * m2 * e[1] + 3. * m1 * e[2] + m0 * e[3])[name]
        uncancelled_M = abs(self.inverse).T @ abs(self.endpoint_metric) @ abs(self.inverse)
        scales = dict(Q=.75+np.linalg.norm(.25*abs(self.K), 2), M=np.linalg.norm(uncancelled_M, 2),
                      R=np.linalg.norm(uncancelled_M / self.D.diagonal()[None, :], 2))
        rounding = 512. * np.finfo(float).eps * max(1., scales[name]) / radius
        return radius**2 * third / 6. + rounding
