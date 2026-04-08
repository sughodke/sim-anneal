"""Image jigsaw puzzle solver using simulated annealing.

Demonstrates the generic Puzzle/Anneal framework with image tiles,
where edge cost = RMSE of pixel values along shared borders.
"""

import scipy.io as sio
import numpy as np
from PIL import Image

import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')

from puzzle import Puzzle
from jax_anneal import JaxAnneal


# --- Image-specific helpers ---

def extract_tiles(image, tile_size):
    """Split an image array into a grid of tile arrays.

    Returns:
        tile_data: dict mapping tile_id -> np.array (tile_size, tile_size, C)
        tile_ids: 2D list of tile IDs in grid order
        h_tiles, w_tiles: grid dimensions
    """
    h_tiles = image.shape[0] // tile_size
    w_tiles = image.shape[1] // tile_size
    tile_data = {}
    tile_ids = []
    tid = 0
    for r in range(h_tiles):
        row = []
        for c in range(w_tiles):
            tile_data[tid] = image[
                r * tile_size:(r + 1) * tile_size,
                c * tile_size:(c + 1) * tile_size,
                :,
            ].astype(np.float64)
            row.append(tid)
            tid += 1
        tile_ids.append(row)
    return tile_data, tile_ids, h_tiles, w_tiles


def image_edge_cost(tile_data):
    """Return an edge cost function that compares pixel borders via RMSE.

    label is (axis, delta) from Puzzle.grid:
      axis 0 = vertical (rows), axis 1 = horizontal (cols)
      delta +1 = neighbor at higher index, -1 = lower index
    """
    def cost(tile_a, orient_a, tile_b, orient_b, label):
        axis, delta = label
        a = tile_data[tile_a]
        b = tile_data[tile_b]
        if axis == 0:
            if delta == 1:  # b is below a
                diff = a[-1, :, :] - b[0, :, :]
            else:  # b is above a
                diff = a[0, :, :] - b[-1, :, :]
        else:
            if delta == 1:  # b is right of a
                diff = a[:, -1, :] - b[:, 0, :]
            else:  # b is left of a
                diff = a[:, 0, :] - b[:, -1, :]
        return np.sqrt((diff ** 2).mean())
    return cost


def render(puzzle, tile_data, tile_size):
    """Reconstruct an image array from the current puzzle state."""
    h = puzzle.grid_shape[0] * tile_size
    w = puzzle.grid_shape[1] * tile_size
    channels = next(iter(tile_data.values())).shape[2]
    img = np.zeros((h, w, channels), dtype=np.uint8)
    for r in range(puzzle.grid_shape[0]):
        for c in range(puzzle.grid_shape[1]):
            tid = puzzle.tiles[(r, c)]
            img[
                r * tile_size:(r + 1) * tile_size,
                c * tile_size:(c + 1) * tile_size,
                :,
            ] = tile_data[tid].astype(np.uint8)
    return img


def main():
    tile_size = 25

    mat = sio.loadmat('shuffledImageEasy.mat')
    shuffled = mat['RGBrearranged'].astype(np.float64)

    tile_data, tile_ids, h, w = extract_tiles(shuffled, tile_size)
    cost_fn = image_edge_cost(tile_data)
    puzzle = Puzzle.grid((h, w), tile_ids, cost_fn)

    solver = JaxAnneal(puzzle)
    solver.run_parallel(start_temp=300, end_temp=1, num_steps=4e6, n_chains=16, sigma=80)

    result = render(puzzle, tile_data, tile_size)
    Image.fromarray(result).save('result.png')
    print("Saved result.png")


if __name__ == '__main__':
    main()
