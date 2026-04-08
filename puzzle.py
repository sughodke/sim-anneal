"""Generic simulated annealing puzzle solver.

The solver operates on a grid of tile IDs and a pluggable edge cost function.
To adapt to a new domain, provide:
  edge_cost(tile_a_id, tile_b_id, direction) -> float
where direction is 'horizontal' (a left of b) or 'vertical' (a above b).
"""

import numpy as np
import time


class Puzzle:
    """A height x width grid of tiles with pluggable edge compatibility."""

    def __init__(self, height, width, tile_ids, edge_cost, swappable=None):
        """
        height, width: grid dimensions in tiles
        tile_ids: flat or 2D sequence of tile identifiers (any hashable)
        edge_cost: callable(tile_a, tile_b, direction) -> float
        swappable: list of (row, col) positions that may be swapped.
                   Defaults to all interior positions (border fixed).
                   Pass 'all' to make every position swappable.
        """
        self.height = height
        self.width = width
        self.tiles = np.array(tile_ids).reshape(height, width)
        self.edge_cost = edge_cost

        if swappable == 'all':
            self.swappable = [
                (r, c) for r in range(height) for c in range(width)
            ]
        elif swappable is not None:
            self.swappable = list(swappable)
        else:
            self.swappable = [
                (r, c)
                for r in range(1, height - 1)
                for c in range(1, width - 1)
            ]

    def swap(self, pos1, pos2):
        r1, c1 = pos1
        r2, c2 = pos2
        self.tiles[r1, c1], self.tiles[r2, c2] = (
            self.tiles[r2, c2], self.tiles[r1, c1],
        )

    def energy_around(self, pos):
        """Sum of edge costs between this tile and its neighbors."""
        r, c = pos
        cost = 0.0
        tid = self.tiles[r, c]
        if c > 0:
            cost += self.edge_cost(self.tiles[r, c - 1], tid, 'horizontal')
        if c < self.width - 1:
            cost += self.edge_cost(tid, self.tiles[r, c + 1], 'horizontal')
        if r > 0:
            cost += self.edge_cost(self.tiles[r - 1, c], tid, 'vertical')
        if r < self.height - 1:
            cost += self.edge_cost(tid, self.tiles[r + 1, c], 'vertical')
        return cost

    def total_energy(self):
        return sum(self.energy_around(pos) for pos in self.swappable)


class Proposal:
    """Biased tile selection — prioritizes high-energy (poorly placed) tiles."""

    def __init__(self, puzzle, recompute_rate=1000, sigma=30):
        self.puzzle = puzzle
        self.sigma = sigma
        self.recompute_rate = recompute_rate
        self.counter = 0
        self.num_tiles = len(puzzle.swappable)
        self._recompute()

    def _recompute(self):
        self.ranked = sorted(
            [(self.puzzle.energy_around(pos), pos) for pos in self.puzzle.swappable],
            reverse=True,
        )

    def get(self):
        self.counter += 1
        if self.counter % self.recompute_rate == 0:
            self.counter = 0
            self._recompute()
        i1 = i2 = 0
        while i1 == i2:
            i1, i2 = np.round(np.clip(
                np.abs(np.random.normal(0, self.sigma, 2)),
                0, self.num_tiles - 1,
            )).astype(int)
        return self.ranked[i1][1], self.ranked[i2][1]


class Anneal:
    """Simulated annealing over tile swaps."""

    def __init__(self, puzzle):
        self.puzzle = puzzle
        self.energy = puzzle.total_energy()

    def _step(self, temperature):
        pos1, pos2 = self.proposal.get()
        old = self.puzzle.energy_around(pos1) + self.puzzle.energy_around(pos2)
        self.puzzle.swap(pos1, pos2)
        new = self.puzzle.energy_around(pos1) + self.puzzle.energy_around(pos2)
        delta = new - old
        if delta > 0:
            if np.random.rand() > np.exp(-delta / temperature):
                self.puzzle.swap(pos1, pos2)  # reject

    def run(self, start_temp, end_temp, num_steps, log_rate=10_000, sigma=30):
        num_steps = int(num_steps)
        self.proposal = Proposal(self.puzzle, sigma=sigma)
        temps = np.logspace(np.log10(start_temp), np.log10(end_temp), num_steps)
        t0 = time.perf_counter()
        for i, temp in enumerate(temps):
            self._step(temp)
            if (i + 1) % log_rate == 0:
                self.energy = self.puzzle.total_energy()
                elapsed = time.perf_counter() - t0
                print(f"step {i+1}/{num_steps}  E={self.energy:.1f}  T={temp:.2f}  t={elapsed:.1f}s")
        self.energy = self.puzzle.total_energy()
