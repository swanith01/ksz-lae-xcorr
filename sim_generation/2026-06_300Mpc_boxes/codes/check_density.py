"""Is density_300.npy the evolved density at each z, or one frozen field?  Read-only."""
import glob, os, sys
import numpy as np
from scipy.integrate import quad

seed_dir = sys.argv[1] if len(sys.argv) > 1 else '/user1/swanith/kSZ2_LAE_project_22Jun2026/seed_6'
want = [5.0, 6.03, 7.08, 9.04, 13.06, 20.31]
dirs = sorted(glob.glob(os.path.join(seed_dir, 'coeval_z*')), key=lambda p: float(p.split('coeval_z')[1]))
zs = np.array([float(p.split('coeval_z')[1]) for p in dirs])
pick = [dirs[int(np.argmin(abs(zs - w)))] for w in want]
pick = sorted(set(pick), key=lambda p: float(p.split('coeval_z')[1]))

Om = 0.3086
E = lambda a: np.sqrt(Om / a**3 + 1 - Om)
D_ = lambda a: 2.5 * Om * E(a) * quad(lambda x: 1 / (x * E(x))**3, 0, a)[0]
D = lambda z: D_(1 / (1 + z)) / D_(1.0)          # linear growth, D(z=0) = 1

print(f'{len(dirs)} snapshot folders in {seed_dir}\n')
ref = None
rows = []
for p in pick:
    z = float(p.split('coeval_z')[1])
    d = np.load(os.path.join(p, 'density_300.npy')).astype(np.float64)
    xH = np.load(os.path.join(p, 'neutral_fraction.npy'))
    v = np.load(os.path.join(p, 'velocity_z.npy'))
    if ref is None:
        ref, zref = d, z
        same = 'reference'
    else:
        diff = np.max(np.abs(d - ref))
        same = 'IDENTICAL to reference' if diff == 0 else f'max|diff| = {diff:.3g}'
    rows.append((z, d.std()))
    print(f'z = {z:7.3f}: density std {d.std():.4f}, min {d.min():8.3f}, cells with delta < -1: {np.mean(d < -1):6.2%}, '
          f'{same}')
    print(f'           [other fields] mean xH {xH.mean():.3f}, peculiar v rms {v.std() * 3.086e19 / (1 + z):6.1f} km/s')
    del d, xH, v

print('\nTest 1 - does the density change with z?  The ionisation and velocity fields do (see [other fields]).')
print('Test 2 - physical bound: a real density has delta >= -1 everywhere (no negative mass).')
print('Test 3 - growth: in linear theory the density std scales with the growth factor D(z):')
z0, s0 = rows[0]
for z, s in rows:
    print(f'   z = {z:6.2f}: measured std ratio {s / s0:.3f}   expected D(z)/D({z0:.2f}) = {D(z) / D(z0):.3f}')
print(f'\nFor reference, the linear field extrapolated to z = 0 scaled to z = 6 would have std {rows[0][1] * D(6.0):.2f} '
      f'(if the saved field is the z = 0 initial conditions).')
