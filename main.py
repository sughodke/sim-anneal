"""Image jigsaw puzzle solver using simulated annealing.

Demonstrates the generic Puzzle/Anneal framework with image tiles,
where edge cost = RMSE of pixel values along shared borders.
"""

import scipy.io as sio
import numpy as np
from PIL import Image

from puzzle import Puzzle, Anneal


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

    axis 0 = vertical (row direction), axis 1 = horizontal (col direction).
    """
    def cost(tile_a, tile_b, axis):
        a = tile_data[tile_a]
        b = tile_data[tile_b]
        if axis == 1:  # horizontal: a is left of b
            diff = a[:, -1, :] - b[:, 0, :]
        else:  # vertical: a is above b
            diff = a[-1, :, :] - b[0, :, :]
        return np.sqrt((diff ** 2).mean())
    return cost


def render(puzzle, tile_data, tile_size):
    """Reconstruct an image array from the current puzzle state."""
    h = puzzle.height * tile_size
    w = puzzle.width * tile_size
    channels = next(iter(tile_data.values())).shape[2]
    img = np.zeros((h, w, channels), dtype=np.uint8)
    for r in range(puzzle.height):
        for c in range(puzzle.width):
            tid = puzzle.tiles[r, c]
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
    puzzle = Puzzle((h, w), tile_ids, cost_fn)

    solver = Anneal(puzzle)
    solver.run(start_temp=160, end_temp=25, num_steps=4e6, sigma=80)

    result = render(puzzle, tile_data, tile_size)
    Image.fromarray(result).save('result.png')
    print("Saved result.png")


if __name__ == '__main__':
    main()
