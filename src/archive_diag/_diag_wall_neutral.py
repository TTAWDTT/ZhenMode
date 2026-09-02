"""FB neutrality: does the CLOSED WALL (mirror ghost _d_dy) break it?

The real solver's lat axis uses _d_dy with mode='edge' (mirror ghost = boundary
value => one-sided diff at the wall). Test 1D closed-wall channel: does the FB
pair amplify vs the periodic version?
"""
import numpy as np
G=9.81; H=4000.0; dt=30.0; ny=120; dy=1.0e5

def d_dy_wall(u):  # mirror-ghost (mode='edge'): ghost=boundary value
    up = np.pad(u, (1,1), mode='edge')
    return (up[2:] - up[:-2]) * (0.5/dy)

def div_cons_wall(v):  # conservative, closed walls (boundary faces closed)
    # face j+1/2 open iff cell j and j+1 both wet; here all-wet so open except
    # the two outer boundary faces (no neighbor). open_j = 1 for j=0..ny-2.
    f = 0.5*(v + np.roll(v,-1))
    f[-1] = 0.0   # boundary face (ny-1 -> 0) closed (wall)
    # div_j = (f_{j+1/2} - f_{j-1/2})/dy; f_{-1/2}=0 (wall)
    div = np.zeros(ny)
    div[1:] = (f[1:] - f[:-1])/dy     # interior faces
    div[0]  = (f[0] - 0.0)/dy         # first cell: in-face = 0 (wall)
    return div

def grad_centered_wall(eta):
    return d_dy_wall(eta)

def energy(eta,v):
    return 0.5*G*H*np.sum(eta**2) + 0.5*H*np.sum(v**2)

def run(divfn, gradfn, label):
    jj=np.arange(ny); eta=0.1*np.sin(np.pi*jj/ny); v=np.zeros(ny)  # seiche, 0 at walls
    e0=energy(eta,v)
    print(f"\n=== {label} ===  E0={e0:.4e}")
    print(f"{'stp':>4} {'E/E0':>10} {'max|eta|':>10} {'max|v|':>10} {'sum_eta':>10}")
    s0=eta.sum()
    for k in range(400):
        div=divfn(v)
        eta = eta - dt*H*div
        g = gradfn(eta)
        v = v - dt*G*g
        if (k+1)%50==0:
            e=energy(eta,v)
            print(f"{k+1:>4} {e/e0:>10.5f} {np.max(np.abs(eta)):>10.4e} {np.max(np.abs(v)):>10.4e} {eta.sum():>10.4e}")

run(d_dy_wall, grad_centered_wall, "centered wall-div + centered wall-grad (solver's _d_dy for both)")
run(div_cons_wall, grad_centered_wall, "conservative wall-div + CENTERED wall-grad (current pairing)")
