"""Generic simulated annealing puzzle solver.

Works on any graph topology (grids, chains, arbitrary graphs) with support
for tile orientations, position-dependent energy, constraints, and partial
occupancy.

To adapt to a new domain, provide:
  edge_cost(tile_a, orient_a, tile_b, orient_b, label) -> float
where label is whatever you store in the adjacency list (axis, direction, etc).
"""

import numpy as np
import time


class Puzzle:
    """Graph-based puzzle with pluggable compatibility functions."""

    def __init__(self, adjacency, tile_ids, edge_cost,
                 position_cost=None, n_orientations=1,
                 swappable=None, constraints=None):
        """
        adjacency: dict mapping position -> [(neighbor_pos, label), ...]
                   Labels are arbitrary and passed through to edge_cost.
        tile_ids: sequence of tile IDs placed at positions in iteration order
                  of adjacency keys. Use None for empty slots.
        edge_cost: callable(tile_a, orient_a, tile_b, orient_b, label) -> float
        position_cost: callable(tile_id, orient, position) -> float (optional)
        n_orientations: number of distinct orientations per tile (default 1)
        swappable: positions that may be modified. Defaults to all.
        constraints: callable(tile_id, orient, position) -> bool (optional).
                     Return False to forbid a placement.
        """
        self.positions = list(adjacency.keys())
        self.adjacency = adjacency
        self.edge_cost = edge_cost
        self.position_cost = position_cost
        self.n_orientations = n_orientations
        self.constraints = constraints

        tile_list = list(tile_ids)
        self.tiles = {}
        self.orients = {}
        for i, pos in enumerate(self.positions):
            self.tiles[pos] = tile_list[i] if i < len(tile_list) else None
            self.orients[pos] = 0

        if swappable is None or swappable == 'all':
            self.swappable = list(self.positions)
        else:
            self.swappable = list(swappable)

    @classmethod
    def grid(cls, shape, tile_ids, edge_cost, border_fixed=True,
             periodic=False, **kwargs):
        """Create a puzzle on a regular n-dimensional grid.

        Labels are (axis, delta) tuples: axis is the dimension, delta is +1
        (neighbor at higher index) or -1 (neighbor at lower index).
        """
        ndim = len(shape)
        adjacency = {}
        for pos in np.ndindex(*shape):
            neighbors = []
            for axis in range(ndim):
                for delta in (-1, +1):
                    idx = pos[axis] + delta
                    if 0 <= idx < shape[axis]:
                        neighbor = list(pos)
                        neighbor[axis] = idx
                        neighbors.append((tuple(neighbor), (axis, delta)))
                    elif periodic:
                        neighbor = list(pos)
                        neighbor[axis] = idx % shape[axis]
                        neighbors.append((tuple(neighbor), (axis, delta)))
            adjacency[pos] = neighbors

        if 'swappable' not in kwargs and border_fixed:
            kwargs['swappable'] = [
                pos for pos in np.ndindex(*shape)
                if all(0 < pos[d] < shape[d] - 1 for d in range(ndim))
            ]

        flat = list(np.array(tile_ids).flatten())
        puzzle = cls(adjacency, flat, edge_cost, **kwargs)
        puzzle.grid_shape = shape
        return puzzle

    @classmethod
    def chain(cls, length, tile_ids, edge_cost, periodic=False, **kwargs):
        """Create a 1D chain (e.g. protein backbone).

        Labels are (0, +1) for forward and (0, -1) for backward.
        All positions are swappable by default.
        """
        adjacency = {}
        for i in range(length):
            neighbors = []
            if i > 0:
                neighbors.append((i - 1, (0, -1)))
            elif periodic:
                neighbors.append((length - 1, (0, -1)))
            if i < length - 1:
                neighbors.append((i + 1, (0, +1)))
            elif periodic:
                neighbors.append((0, (0, +1)))
            adjacency[i] = neighbors
        return cls(adjacency, list(tile_ids), edge_cost, **kwargs)

    def swap(self, pos1, pos2):
        self.tiles[pos1], self.tiles[pos2] = self.tiles[pos2], self.tiles[pos1]
        self.orients[pos1], self.orients[pos2] = self.orients[pos2], self.orients[pos1]

    def rotate(self, pos, orientation):
        self.orients[pos] = orientation

    def energy_around(self, pos):
        """Energy contribution of the tile at pos (edges + position cost)."""
        tid = self.tiles[pos]
        if tid is None:
            return 0.0
        orient = self.orients[pos]
        cost = 0.0
        if self.position_cost is not None:
            cost += self.position_cost(tid, orient, pos)
        for neighbor, label in self.adjacency[pos]:
            ntid = self.tiles[neighbor]
            if ntid is None:
                continue
            cost += self.edge_cost(tid, orient, ntid, self.orients[neighbor], label)
        return cost

    def total_energy(self):
        return sum(self.energy_around(pos) for pos in self.swappable)

    def check_constraint(self, tile_id, orientation, position):
        if self.constraints is None:
            return True
        return self.constraints(tile_id, orientation, position)


class Proposal:
    """Generates moves: swaps and (optionally) rotations, biased toward
    high-energy tiles."""

    def __init__(self, puzzle, recompute_rate=1000, sigma=30, rotate_fraction=0.3):
        self.puzzle = puzzle
        self.sigma = sigma
        self.recompute_rate = recompute_rate
        self.rotate_fraction = rotate_fraction if puzzle.n_orientations > 1 else 0.0
        self.counter = 0
        self.num_tiles = len(puzzle.swappable)
        self._recompute()

    def _recompute(self):
        self.ranked = sorted(
            [(self.puzzle.energy_around(pos), pos) for pos in self.puzzle.swappable],
            reverse=True,
        )

    def _pick_index(self):
        return int(np.round(np.clip(
            abs(np.random.normal(0, self.sigma)),
            0, self.num_tiles - 1,
        )))

    def get(self):
        """Return ('swap', pos1, pos2) or ('rotate', pos, new_orient)."""
        self.counter += 1
        if self.counter % self.recompute_rate == 0:
            self.counter = 0
            self._recompute()

        if np.random.rand() < self.rotate_fraction:
            idx = self._pick_index()
            pos = self.ranked[idx][1]
            return ('rotate', pos, np.random.randint(0, self.puzzle.n_orientations))

        i1 = i2 = 0
        while i1 == i2:
            i1, i2 = np.round(np.clip(
                np.abs(np.random.normal(0, self.sigma, 2)),
                0, self.num_tiles - 1,
            )).astype(int)
        return ('swap', self.ranked[i1][1], self.ranked[i2][1])


class Anneal:
    """Simulated annealing over tile moves (swaps and rotations)."""

    def __init__(self, puzzle):
        self.puzzle = puzzle
        self.energy = puzzle.total_energy()

    def _step(self, temperature):
        move = self.proposal.get()

        if move[0] == 'swap':
            pos1, pos2 = move[1], move[2]
            t1, o1 = self.puzzle.tiles[pos1], self.puzzle.orients[pos1]
            t2, o2 = self.puzzle.tiles[pos2], self.puzzle.orients[pos2]
            if t1 is not None and not self.puzzle.check_constraint(t1, o1, pos2):
                return
            if t2 is not None and not self.puzzle.check_constraint(t2, o2, pos1):
                return
            affected = [pos1, pos2]
        else:
            pos, new_orient = move[1], move[2]
            tid = self.puzzle.tiles[pos]
            if tid is None:
                return
            if not self.puzzle.check_constraint(tid, new_orient, pos):
                return
            old_orient = self.puzzle.orients[pos]
            affected = [pos]

        old = sum(self.puzzle.energy_around(p) for p in affected)

        if move[0] == 'swap':
            self.puzzle.swap(pos1, pos2)
        else:
            self.puzzle.rotate(pos, new_orient)

        new = sum(self.puzzle.energy_around(p) for p in affected)
        delta = new - old

        if delta > 0 and np.random.rand() > np.exp(-delta / temperature):
            if move[0] == 'swap':
                self.puzzle.swap(pos1, pos2)
            else:
                self.puzzle.rotate(pos, old_orient)

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
