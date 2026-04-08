import numpy as np
import pytest
from puzzle import Puzzle, Anneal


class Test2D:
    """4x4 grid, interior tiles shuffled, must reach sequential order."""

    def setup_method(self):
        self.shape = (4, 4)
        grid = np.arange(16).reshape(self.shape).copy()
        interior = [(r, c) for r in range(1, 3) for c in range(1, 3)]
        vals = [grid[r, c] for r, c in interior]
        np.random.default_rng(42).shuffle(vals)
        for (r, c), v in zip(interior, vals):
            grid[r, c] = v

        strides = [self.shape[1], 1]

        def cost(a, oa, b, ob, label):
            axis, delta = label
            return 0.0 if b - a == delta * strides[axis] else float(abs(b - a))

        self.puzzle = Puzzle.grid(self.shape, grid.flatten(), cost)

    def test_solves(self):
        Anneal(self.puzzle).run(50, 0.1, 50_000, log_rate=100_000, sigma=5)
        assert self.puzzle.total_energy() == 0.0

    def test_tiles_in_order(self):
        Anneal(self.puzzle).run(50, 0.1, 50_000, log_rate=100_000, sigma=5)
        for r in range(4):
            for c in range(4):
                assert self.puzzle.tiles[(r, c)] == r * 4 + c


class Test3D:
    """4x4x4 grid, interior tiles shuffled."""

    def setup_method(self):
        self.shape = (4, 4, 4)
        grid = np.arange(64).reshape(self.shape).copy()
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

        self.puzzle = Puzzle.grid(self.shape, grid.flatten(), cost)

    def test_solves(self):
        Anneal(self.puzzle).run(100, 0.1, 200_000, log_rate=500_000, sigma=10)
        assert self.puzzle.total_energy() == 0.0


class TestChain:
    """8-element chain, fragments must end up in sequence."""

    def setup_method(self):
        tiles = list(range(8))
        np.random.default_rng(42).shuffle(tiles)

        def cost(a, oa, b, ob, label):
            _, delta = label
            return 0.0 if b - a == delta else float(abs(b - a))

        self.puzzle = Puzzle.chain(8, tiles, cost)

    def test_solves(self):
        Anneal(self.puzzle).run(50, 0.01, 50_000, log_rate=100_000, sigma=5)
        assert self.puzzle.total_energy() == 0.0

    def test_order(self):
        Anneal(self.puzzle).run(50, 0.01, 50_000, log_rate=100_000, sigma=5)
        for i in range(8):
            assert self.puzzle.tiles[i] == i


class TestPeriodicChain:
    """Ring of 6 tiles in cyclic order."""

    def setup_method(self):
        self.n = 6
        tiles = list(range(self.n))
        np.random.default_rng(99).shuffle(tiles)

        n = self.n

        def cost(a, oa, b, ob, label):
            _, delta = label
            expected = (a + delta) % n
            return 0.0 if b == expected else float(min(abs(b - a), n - abs(b - a)))

        self.puzzle = Puzzle.chain(self.n, tiles, cost, periodic=True)

    def test_solves(self):
        Anneal(self.puzzle).run(50, 0.01, 50_000, log_rate=100_000, sigma=5)
        assert self.puzzle.total_energy() == 0.0


class TestOrientation:
    """Chain of 4 tiles with 2 orientations each. Solver must find correct
    placement AND orientation."""

    def setup_method(self):
        tiles = [3, 1, 0, 2]

        def cost(a, oa, b, ob, label):
            _, delta = label
            return 0.0 if b - a == delta else float(abs(b - a))

        def pos_cost(tid, orient, pos):
            correct = tid % 2
            return 0.0 if orient == correct else 5.0

        self.puzzle = Puzzle.chain(4, tiles, cost,
                                  n_orientations=2, position_cost=pos_cost)

    def test_solves(self):
        Anneal(self.puzzle).run(50, 0.01, 100_000, log_rate=200_000, sigma=5)
        assert self.puzzle.total_energy() == 0.0

    def test_orientations_correct(self):
        Anneal(self.puzzle).run(50, 0.01, 100_000, log_rate=200_000, sigma=5)
        for pos in range(4):
            tid = self.puzzle.tiles[pos]
            assert self.puzzle.orients[pos] == tid % 2


class TestPositionCost:
    """Tiles with preferred positions (self-energy only, no edge cost)."""

    def setup_method(self):
        tiles = [2, 0, 3, 1]

        def edge_cost(a, oa, b, ob, label):
            return 0.0

        def pos_cost(tid, orient, pos):
            return 0.0 if tid == pos else 5.0

        self.puzzle = Puzzle.chain(4, tiles, edge_cost, position_cost=pos_cost)

    def test_solves(self):
        Anneal(self.puzzle).run(50, 0.01, 50_000, log_rate=100_000, sigma=5)
        for pos in range(4):
            assert self.puzzle.tiles[pos] == pos


class TestConstraints:
    """4-element chain where even tiles must go in even positions."""

    def setup_method(self):
        tiles = [3, 2, 1, 0]

        def cost(a, oa, b, ob, label):
            _, delta = label
            return 0.0 if b - a == delta else float(abs(b - a))

        def constraint(tid, orient, pos):
            return tid % 2 == pos % 2

        self.puzzle = Puzzle.chain(4, tiles, cost, constraints=constraint)

    def test_constraints_respected(self):
        Anneal(self.puzzle).run(50, 0.01, 50_000, log_rate=100_000, sigma=5)
        for pos in range(4):
            tid = self.puzzle.tiles[pos]
            assert tid % 2 == pos % 2

    def test_solves(self):
        Anneal(self.puzzle).run(50, 0.01, 50_000, log_rate=100_000, sigma=5)
        assert self.puzzle.total_energy() == 0.0


class TestPartialOccupancy:
    """5-position chain with only 3 tiles and 2 empty slots."""

    def setup_method(self):
        # tiles at positions 0,1,2; None at 3,4
        tiles = [2, None, 0, None, 1]

        def cost(a, oa, b, ob, label):
            _, delta = label
            return 0.0 if b - a == delta else float(abs(b - a))

        self.puzzle = Puzzle.chain(5, tiles, cost)

    def test_none_tiles_stay(self):
        Anneal(self.puzzle).run(50, 0.01, 50_000, log_rate=100_000, sigma=5)
        # solver can move Nones around, but energy should be finite
        none_count = sum(1 for pos in range(5) if self.puzzle.tiles[pos] is None)
        tile_count = sum(1 for pos in range(5) if self.puzzle.tiles[pos] is not None)
        assert none_count == 2
        assert tile_count == 3


class TestArbitraryGraph:
    """Triangle graph — 3 nodes, 3 edges, custom adjacency."""

    def setup_method(self):
        adjacency = {
            'A': [('B', 'ab'), ('C', 'ac')],
            'B': [('A', 'ab'), ('C', 'bc')],
            'C': [('A', 'ac'), ('B', 'bc')],
        }
        # tile 0 at A, tile 1 at B, tile 2 at C is optimal
        tiles = [2, 0, 1]  # shuffled

        def cost(a, oa, b, ob, label):
            pairs = {'ab': (0, 1), 'ac': (0, 2), 'bc': (1, 2)}
            expected = pairs[label]
            return 0.0 if set([a, b]) == set(expected) else 5.0

        self.puzzle = Puzzle(adjacency, tiles, cost)

    def test_solves(self):
        Anneal(self.puzzle).run(50, 0.01, 50_000, log_rate=100_000, sigma=3)
        assert self.puzzle.total_energy() == 0.0
