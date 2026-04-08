"""3D voxel cat puzzle: reassemble a cat from shuffled colored blocks.

A 3D voxel model of a cat is built on a grid, with each filled voxel
assigned a color (orange body, white belly, dark ears, green eyes, pink
nose). The colored blocks are shuffled among the cat positions and the
solver must reconstruct the cat using color continuity in 3D.

Uses:
  - 3D grid topology (Puzzle.grid with 3 dimensions)
  - Partial occupancy (None for air voxels)
  - Edge cost from RGB color distance along all 3 axes

Usage:
    uv run python examples/plusplus_voxel_cat/main.py
"""

import sys, os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
from PIL import Image
from puzzle import Puzzle
from jax_anneal import JaxAnneal

# Colors (RGB)
ORANGE = np.array([220, 140, 40], dtype=np.float64)
DARK_ORANGE = np.array([180, 100, 20], dtype=np.float64)
WHITE = np.array([240, 235, 230], dtype=np.float64)
BLACK = np.array([30, 30, 35], dtype=np.float64)
PINK = np.array([240, 150, 150], dtype=np.float64)
GREEN = np.array([80, 200, 80], dtype=np.float64)
TAIL_TIP = np.array([250, 250, 240], dtype=np.float64)


def build_cat(shape=(14, 10, 16)):
    """Generate a 3D voxel cat.

    Axes: x=width, y=depth, z=height (z=0 is ground).
    Returns a dict mapping (x, y, z) -> RGB color, or None for air.
    """
    W, D, H = shape
    voxels = {}

    def fill(x, y, z, color):
        if 0 <= x < W and 0 <= y < D and 0 <= z < H:
            # Add slight per-voxel variation for distinguishability
            voxels[(x, y, z)] = color.copy()

    def box(x0, x1, y0, y1, z0, z1, color):
        for x in range(x0, x1):
            for y in range(y0, y1):
                for z in range(z0, z1):
                    fill(x, y, z, color)

    cx = W // 2

    # Legs (4 columns)
    for lx, ly in [(cx - 3, 2), (cx + 2, 2), (cx - 3, 6), (cx + 2, 6)]:
        box(lx, lx + 2, ly, ly + 2, 0, 4, ORANGE)

    # Body
    box(cx - 4, cx + 4, 1, 8, 4, 9, ORANGE)
    # Belly (white underside)
    box(cx - 2, cx + 2, 2, 7, 4, 5, WHITE)

    # Neck
    box(cx - 3, cx + 3, 2, 7, 9, 10, ORANGE)

    # Head
    box(cx - 3, cx + 3, 1, 7, 10, 14, ORANGE)
    # Face (front)
    box(cx - 2, cx + 2, 1, 2, 11, 13, WHITE)
    # Eyes
    fill(cx - 2, 1, 12, GREEN)
    fill(cx + 1, 1, 12, GREEN)
    # Nose
    fill(cx, 1, 11, PINK)
    # Mouth
    fill(cx - 1, 1, 11, BLACK)
    fill(cx + 1, 1, 11, BLACK) # whisker dots

    # Ears
    for ex in [cx - 3, cx + 2]:
        fill(ex, 3, 14, DARK_ORANGE)
        fill(ex + 1, 3, 14, DARK_ORANGE)
        fill(ex, 4, 14, DARK_ORANGE)
        fill(ex + 1, 4, 14, DARK_ORANGE)
        fill(ex, 3, 15, DARK_ORANGE)
        fill(ex + 1, 4, 15, DARK_ORANGE)
        # Inner ear
        fill(ex + 1, 3, 14, PINK)

    # Tail (curves up from back)
    for i in range(6):
        tz = 7 + i
        ty = 8 if i < 3 else 9
        fill(cx, ty, tz, ORANGE if i < 4 else TAIL_TIP)

    # Add per-voxel noise
    rng = np.random.default_rng(42)
    for pos in voxels:
        voxels[pos] = np.clip(voxels[pos] + rng.normal(0, 5, 3), 0, 255)

    return voxels, shape


def render_slices(voxels, shape, tile_size=16):
    """Render Z-slices of the voxel model as a grid image."""
    W, D, H = shape
    cols = 4
    rows = (H + cols - 1) // cols
    gap = 2

    img_w = cols * W * tile_size + (cols - 1) * gap
    img_h = rows * D * tile_size + (rows - 1) * gap
    img = np.full((img_h, img_w, 3), 200, dtype=np.uint8)  # gray background

    for z in range(H):
        col = z % cols
        row = z // cols
        ox = col * (W * tile_size + gap)
        oy = row * (D * tile_size + gap)

        for x in range(W):
            for y in range(D):
                color = voxels.get((x, y, z))
                if color is not None:
                    px = ox + x * tile_size
                    py = oy + y * tile_size
                    img[py:py + tile_size, px:px + tile_size] = color.astype(np.uint8)
                else:
                    px = ox + x * tile_size
                    py = oy + y * tile_size
                    img[py:py + tile_size, px:px + tile_size] = [230, 230, 230]

    return img


def export_ply(voxels, path):
    """Export voxels as a colored PLY mesh.

    Only emits exposed faces (where a voxel borders air). Each quad face
    is split into two triangles with the voxel's RGB color.
    """
    # 6 face directions: (axis_offset, vertices_template)
    # Each face is 4 vertices forming a quad on one side of the unit cube
    face_defs = [
        # (dx,dy,dz) neighbor offset, then 4 corner vertices of the face
        ((-1, 0, 0), [(0,0,0),(0,1,0),(0,1,1),(0,0,1)]),  # -X
        ((+1, 0, 0), [(1,0,0),(1,0,1),(1,1,1),(1,1,0)]),  # +X
        ((0, -1, 0), [(0,0,0),(0,0,1),(1,0,1),(1,0,0)]),  # -Y
        ((0, +1, 0), [(0,1,0),(1,1,0),(1,1,1),(0,1,1)]),  # +Y
        ((0, 0, -1), [(0,0,0),(1,0,0),(1,1,0),(0,1,0)]),  # -Z
        ((0, 0, +1), [(0,0,1),(0,1,1),(1,1,1),(1,0,1)]),  # +Z
    ]

    vertices = []
    faces = []  # (v0, v1, v2, r, g, b)

    for (x, y, z), color in voxels.items():
        r, g, b = int(color[0]), int(color[1]), int(color[2])
        for (dx, dy, dz), corners in face_defs:
            neighbor = (x + dx, y + dy, z + dz)
            if neighbor in voxels:
                continue  # face is hidden
            # Emit quad as 2 triangles
            base = len(vertices)
            for cx, cy, cz in corners:
                vertices.append((x + cx, y + cy, z + cz))
            faces.append((base, base+1, base+2, r, g, b))
            faces.append((base, base+2, base+3, r, g, b))

    with open(path, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for vx, vy, vz in vertices:
            f.write(f"{vx} {vy} {vz}\n")
        for v0, v1, v2, r, g, b in faces:
            f.write(f"3 {v0} {v1} {v2} {r} {g} {b}\n")


def main():
    shape = (14, 10, 16)
    voxels, shape = build_cat(shape)
    n_filled = len(voxels)

    here = os.path.dirname(__file__)
    print(f"3D Voxel Cat: {shape[0]}x{shape[1]}x{shape[2]} grid")
    print(f"Filled voxels: {n_filled}")

    # Save target
    target_img = render_slices(voxels, shape)
    Image.fromarray(target_img).save(os.path.join(here, 'target.png'))
    export_ply(voxels, os.path.join(here, 'target.ply'))
    print("Saved target.png + target.ply")

    # Build tile data: map tile_id -> color
    filled_positions = sorted(voxels.keys())
    tile_data = {}
    for tid, pos in enumerate(filled_positions):
        tile_data[tid] = voxels[pos]

    # Build flat tile_ids array for full grid (None for air)
    pos_to_tid = {pos: tid for tid, pos in enumerate(filled_positions)}
    all_tile_ids = []
    all_positions = list(np.ndindex(*shape))
    for pos in all_positions:
        if pos in voxels:
            all_tile_ids.append(pos_to_tid[pos])
        else:
            all_tile_ids.append(None)

    # Edge cost: RGB distance
    def edge_cost(a, orient_a, b, orient_b, label):
        diff = tile_data[a] - tile_data[b]
        return np.sqrt((diff ** 2).mean())

    # Only filled positions are swappable
    swappable = [pos for pos in all_positions if pos in voxels]

    puzzle = Puzzle.grid(shape, all_tile_ids, edge_cost, border_fixed=False,
                         swappable=swappable)
    print(f"Swappable: {len(swappable)}")

    # Shuffle filled tiles among filled positions
    rng = np.random.default_rng(123)
    filled_tids = [puzzle.tiles[pos] for pos in swappable]
    rng.shuffle(filled_tids)
    for pos, tid in zip(swappable, filled_tids):
        puzzle.tiles[pos] = tid

    print()

    solver = JaxAnneal(puzzle)
    solver.run_parallel(start_temp=200, end_temp=1, num_steps=4e6,
                        n_chains=16, log_rate=400_000, sigma=40)

    # Render result
    result_voxels = {}
    for pos in all_positions:
        tid = puzzle.tiles.get(pos)
        if tid is not None:
            result_voxels[pos] = tile_data[tid]

    result_img = render_slices(result_voxels, shape)
    Image.fromarray(result_img).save(os.path.join(here, 'result.png'))
    export_ply(result_voxels, os.path.join(here, 'result.ply'))
    print(f"\nSaved result.png + result.ply")


if __name__ == '__main__':
    main()
