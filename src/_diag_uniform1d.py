"""Minimal numpy FB neutrality test on a uniform 1D periodic channel.

Replicates _free_surface_step_fd math EXACTLY but on a clean grid:
  - all-wet, periodic lon, closed walls OFF (use periodic in lat too via roll)
  - uniform dx, r_bot=0, no cap, no sponge, no forcing
Tests: is the FB pair neutral (|lambda|=1) here?
Then: same but with the conservative div + adjoint grad.
Then: same with closed walls (mirror ghost) instead of periodic.
"""
import numpy as np

G = 9.81; H = 4000.0; dt = 30.0   # dt_half
nx = 360; dx = 1.0e5

def d_dx_roll(u):  # centered periodic (the solver's _d_dx)
    return (np.roll(u,-1) - np.roll(u,1)) * (0.5/dx)

def div_cons(u):   # conservative flux-form, all-wet (open=1 everywhere periodic)
    f = 0.5*(u + np.roll(u,-1))
    return (f - np.roll(f,1))/dx

def grad_centered(eta):  # centered (adjoint of div_cons on uniform all-wet? check)
    return d_dx_roll(eta)

def grad_adjoint(eta):   # exact adjoint of div_cons (face difference form)
    return 0.5*((np.roll(eta,-1)-eta) + (eta-np.roll(eta,1)))/dx

def energy(eta,u):
    return 0.5*G*H*np.sum(eta**2) + 0.5*H*np.sum(u**2)

def run(divfn, gradfn, label):
    ii=np.arange(nx); eta=0.1*np.sin(2*np.pi*ii/nx); u=np.zeros(nx)
    e0=energy(eta,u); prev=e0
    print(f"\n=== {label} ===")
    print(f"{'stp':>4} {'E/E0':>10} {'max|eta|':>10} {'max|u|':>10}")
    for k in range(300):
        div=divfn(u)
        eta = eta - dt*H*div
        g = gradfn(eta)
        u = u - dt*G*g
        if (k+1)%50==0:
            e=energy(eta,u)
            print(f"{k+1:>4} {e/e0:>10.5f} {np.max(np.abs(eta)):>10.4e} {np.max(np.abs(u)):>10.4e}")
            prev=e

run(d_dx_roll, grad_centered, "centered div + centered grad (solver's _d_dx for both)")
run(div_cons, grad_centered, "conservative div + CENTERED grad (current solver pairing)")
run(div_cons, grad_adjoint,  "conservative div + ADJOINT grad (the fix)")
