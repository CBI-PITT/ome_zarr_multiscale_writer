import numpy as np
import math

# ---------- Helpers ----------
def ceil_div(a, b):  # integer ceil
    return -(-a // b)

def ds2_mean_uint16(img: np.ndarray) -> np.ndarray:
    y, x = img.shape
    y2 = y - (y & 1); x2 = x - (x & 1)
    out = img[:y2:2, :x2:2].astype(np.uint32)
    out += img[1:y2:2, :x2:2].astype(np.uint32)
    out += img[:y2:2, 1:x2:2].astype(np.uint32)
    out += img[1:y2:2, 1:x2:2].astype(np.uint32)
    out += 2 # +2 to mean round divide by 4
    out[:] = out >> 2
    # pad edge by replication if odd dims:
    if y & 1: out = np.vstack([out, out[-1:]])
    if x & 1: out = np.hstack([out, out[:, -1:]])
    return out.astype(np.uint16)

def dsZ2_mean_uint16(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Mean of two uint16 slices -> uint16."""
    out = a.astype(np.uint32)
    out += b.astype(np.uint32)
    out += 1 # +1 for mean round divide by 2
    out[:] = out >> 1
    return out.astype(np.uint16)

def infer_n_levels(y, x, z_estimate, min_dim=256):
    """Stop when any axis would shrink below min_dim (spatial) or z_estimate//2**L < 1."""
    levels = 1
    while (min(y, x) // (2 ** levels) >= min_dim) and (z_estimate // (2 ** levels) >= 1):
        levels += 1
    return levels

def compute_xy_only_levels(voxel_size):
    """Return how many leading pyramid levels should be XY-only (no Z decimation),
    so that XY spacing never exceeds Z spacing."""
    dz, dy, dx = map(float, voxel_size)
    # how many 2x steps can XY take without exceeding dz?
    ky = 0 if dy <= 0 else max(0, math.floor(math.log2(dz / dy)))
    kx = 0 if dx <= 0 else max(0, math.floor(math.log2(dz / dx)))
    return int(max(0, min(ky, kx)))  # lockstep XY downsampling

def level_factors(level: int, xy_levels: int):
    """Per-level physical scaling factors relative to level 0 for (z,y,x)."""
    if level <= xy_levels:
        zf = 1
    else:
        zf = 2 ** (level - xy_levels)
    yf = 2 ** level
    xf = 2 ** level
    return zf, yf, xf

def plan_levels(z_estimate, y, x, xy_levels, min_dim=256):
    """Total levels given XY-only prelude, then 3D, stopping when
       min(y_l, x_l) < min_dim or z_l < 1."""
    L = 1
    while True:
        zf, yf, xf = level_factors(L, xy_levels)
        y_l = ceil_div(y, yf)
        x_l = ceil_div(x, xf)
        z_l = ceil_div(z_estimate, zf)
        if min(y_l, x_l) < min_dim or z_l < 1:
            break
        L += 1
    return L