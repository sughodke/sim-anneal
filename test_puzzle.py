import numpy as np
import pytest
from puzzle import Puzzle, Anneal


def sequential_cost(tile_a, tile_b, axis):
    """Tiles want to be in sequential order along every axis.

    Cost is 0 when tile_b == tile_a + stride for that axis, nonzero otherwise.
    We embed the expected stride in the tile IDs themselves: for a grid of
    shape S, tile at position (i, j, ...) has ID = np.ravel_multi_index((i, j, ...), S).
    """
    return 0.0 if tile_b == tile_a + 1 else abs(tile_b - tile_a)


class Test2D:
    """Solve a small 2D puzzle where tiles should end up in row-major order."""

    def setup_method(self):
        self.shape = (4, 4)
        # solved state: 0..15 in row-major order
        solved = np.arange(16)
        # shuffle only interior tiles (border stays fixed)
        grid = solved.reshape(self.shape).copy()
        interior = [(r, c) for r in range(1, 3) for c in range(1, 3)]
        interior_vals = [grid[r, c] for r, c in interior]
        rng = np.random.default_rng(42)
        rng.shuffle(interior_vals)
        for (r, c), v in zip(interior, interior_vals):
            grid[r, c] = v

        def cost(a, b, axis):
            stride = self.shape[1] if axis == 0 else 1
            return 0.0 if b == a + stride else float(abs(b - a))

        self.cost = cost
        self.puzzle = Puzzle(self.shape, grid.flatten(), cost)

    def test_solves(self):
        solver = Anneal(self.puzzle)
        solver.run(start_temp=50, end_temp=0.1, num_steps=50_000, log_rate=100_000, sigma=5)
        assert solver.energy == 0.0

    def test_tiles_in_order(self):
        solver = Anneal(self.puzzle)
        solver.run(start_temp=50, end_temp=0.1, num_steps=50_000, log_rate=100_000, sigma=5)
        expected = np.arange(16).reshape(self.shape)
        np.testing.assert_array_equal(self.puzzle.tiles, expected)


class Test3D:
    """Solve a small 3D puzzle where tiles should end up in row-major order."""

    def setup_method(self):
        self.shape = (4, 4, 4)
        n = 64
        solved = np.arange(n)
        grid = solved.reshape(self.shape).copy()
        # shuffle interior positions
        interior = [
            (x, y, z)
            for x in range(1, 3) for y in range(1, 3) for z in range(1, 3)
        ]
        interior_vals = [grid[pos] for pos in interior]
        rng = np.random.default_rng(123)
        rng.shuffle(interior_vals)
        for pos, v in zip(interior, interior_vals):
            grid[pos] = v

        strides = [self.shape[1] * self.shape[2], self.shape[2], 1]

        def cost(a, b, axis):
            return 0.0 if b == a + strides[axis] else float(abs(b - a))

        self.cost = cost
        self.puzzle = Puzzle(self.shape, grid.flatten(), cost)

    def test_solves(self):
        solver = Anneal(self.puzzle)
        solver.run(start_temp=100, end_temp=0.1, num_steps=200_000, log_rate=500_000, sigma=10)
        assert solver.energy == 0.0

    def test_tiles_in_order(self):
        solver = Anneal(self.puzzle)
        solver.run(start_temp=100, end_temp=0.1, num_steps=200_000, log_rate=500_000, sigma=10)
        expected = np.arange(64).reshape(self.shape)
        np.testing.assert_array_equal(self.puzzle.tiles, expected)
