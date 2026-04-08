import numpy as np
import pytest
from puzzle import Puzzle
from jax_anneal import JaxAnneal


class TestJax2D:
    """4x4 grid via JAX solver."""

    def setup_method(self):
        shape = (4, 4)
        grid = np.arange(16).reshape(shape).copy()
        interior = [(r, c) for r in range(1, 3) for c in range(1, 3)]
        vals = [grid[r, c] for r, c in interior]
        np.random.default_rng(42).shuffle(vals)
        for (r, c), v in zip(interior, vals):
            grid[r, c] = v

        strides = [shape[1], 1]

        def cost(a, oa, b, ob, label):
            axis, delta = label
            return 0.0 if b - a == delta * strides[axis] else float(abs(b - a))

        self.puzzle = Puzzle.grid(shape, grid.flatten(), cost)

    def test_solves(self):
        solver = JaxAnneal(self.puzzle)
        solver.run(50, 0.1, 50_000, log_rate=50_000, seed=42)
        assert solver.energy == 0.0


class TestJax3D:
    """4x4x4 grid via JAX solver."""

    def setup_method(self):
        shape = (4, 4, 4)
        grid = np.arange(64).reshape(shape).copy()
        interior = [
            (x, y, z)
            for x in range(1, 3) for y in range(1, 3) for z in range(1, 3)
        ]
        vals = [grid[pos] for pos in interior]
        np.random.default_rng(123).shuffle(vals)
        for pos, v in zip(interior, vals):
            grid[pos] = v

        strides = [16, 4, 1]

        def cost(a, oa, b, ob, label):
            axis, delta = label
            return 0.0 if b - a == delta * strides[axis] else float(abs(b - a))

        self.puzzle = Puzzle.grid(shape, grid.flatten(), cost)

    def test_solves(self):
        solver = JaxAnneal(self.puzzle)
        solver.run(100, 0.1, 200_000, log_rate=200_000, seed=123)
        assert solver.energy == 0.0


class TestJaxChain:
    """8-element chain via JAX solver."""

    def setup_method(self):
        tiles = list(range(8))
        np.random.default_rng(42).shuffle(tiles)

        def cost(a, oa, b, ob, label):
            _, delta = label
            return 0.0 if b - a == delta else float(abs(b - a))

        self.puzzle = Puzzle.chain(8, tiles, cost)

    def test_solves(self):
        solver = JaxAnneal(self.puzzle)
        solver.run(50, 0.01, 50_000, log_rate=50_000, seed=42)
        assert solver.energy == 0.0


class TestJaxOrientation:
    """Chain with orientations via JAX solver."""

    def setup_method(self):
        tiles = [3, 1, 0, 2]

        def cost(a, oa, b, ob, label):
            _, delta = label
            return 0.0 if b - a == delta else float(abs(b - a))

        def pos_cost(tid, orient, pos):
            return 0.0 if orient == tid % 2 else 5.0

        self.puzzle = Puzzle.chain(4, tiles, cost,
                                  n_orientations=2, position_cost=pos_cost)

    def test_solves(self):
        solver = JaxAnneal(self.puzzle)
        solver.run(50, 0.01, 100_000, log_rate=100_000, seed=42)
        assert solver.energy == 0.0

    def test_orientations_correct(self):
        solver = JaxAnneal(self.puzzle)
        solver.run(50, 0.01, 100_000, log_rate=100_000, seed=42)
        for pos in range(4):
            tid = self.puzzle.tiles[pos]
            assert self.puzzle.orients[pos] == tid % 2


class TestJaxConstraints:
    """Chain with constraints via JAX solver."""

    def setup_method(self):
        tiles = [3, 2, 1, 0]

        def cost(a, oa, b, ob, label):
            _, delta = label
            return 0.0 if b - a == delta else float(abs(b - a))

        def constraint(tid, orient, pos):
            return tid % 2 == pos % 2

        self.puzzle = Puzzle.chain(4, tiles, cost, constraints=constraint)

    def test_constraints_respected(self):
        solver = JaxAnneal(self.puzzle)
        solver.run(50, 0.01, 50_000, log_rate=50_000, seed=42)
        for pos in range(4):
            tid = self.puzzle.tiles[pos]
            assert tid % 2 == pos % 2


class TestJaxParallel:
    """Parallel chains find the solution."""

    def setup_method(self):
        tiles = list(range(8))
        np.random.default_rng(99).shuffle(tiles)

        def cost(a, oa, b, ob, label):
            _, delta = label
            return 0.0 if b - a == delta else float(abs(b - a))

        self.puzzle = Puzzle.chain(8, tiles, cost)

    def test_parallel_solves(self):
        solver = JaxAnneal(self.puzzle)
        solver.run_parallel(50, 0.01, 20_000, n_chains=8, seed=42)
        assert solver.energy == 0.0
