"""JAX-accelerated simulated annealing solver.

Compiles the SA inner loop via XLA and supports parallel chains via vmap.
Use with Puzzle from puzzle.py — same setup, faster execution.

    from puzzle import Puzzle
    from jax_anneal import JaxAnneal

    puzzle = Puzzle.chain(100, tiles, cost)
    solver = JaxAnneal(puzzle)
    solver.run(100, 1, 1e6)                          # single chain
    solver.run_parallel(100, 1, 250_000, n_chains=64) # parallel chains
"""

import jax
import jax.numpy as jnp
from jax import lax
import numpy as np
import time


class JaxAnneal:
    """JIT-compiled simulated annealing with optional parallel chains."""

    def __init__(self, puzzle):
        self.puzzle = puzzle
        self._compile(puzzle)

    def _compile(self, puzzle):
        """Convert puzzle graph to flat arrays and precompute cost matrices."""
        pos_list = list(puzzle.positions)
        pos_to_idx = {p: i for i, p in enumerate(pos_list)}
        n_pos = len(pos_list)

        swap_idx = jnp.array([pos_to_idx[p] for p in puzzle.swappable], dtype=jnp.int32)
        n_swap = len(puzzle.swappable)

        # Tile ID mapping (None -> EMPTY sentinel)
        tile_set = sorted(set(t for t in puzzle.tiles.values() if t is not None))
        tile_to_idx = {t: i for i, t in enumerate(tile_set)}
        n_tiles = len(tile_set)
        EMPTY = n_tiles

        # Label mapping
        all_labels = sorted(
            {l for nbrs in puzzle.adjacency.values() for _, l in nbrs}, key=str
        )
        label_to_idx = {l: i for i, l in enumerate(all_labels)}
        n_labels = max(len(all_labels), 1)

        # Adjacency arrays (padded to max_neighbors)
        max_nbrs = max(len(nbrs) for nbrs in puzzle.adjacency.values())
        adj_nbrs = np.zeros((n_pos, max_nbrs), dtype=np.int32)
        adj_lbls = np.zeros((n_pos, max_nbrs), dtype=np.int32)
        adj_mask = np.zeros((n_pos, max_nbrs), dtype=bool)
        for pos in pos_list:
            pi = pos_to_idx[pos]
            for j, (nbr, lbl) in enumerate(puzzle.adjacency[pos]):
                adj_nbrs[pi, j] = pos_to_idx[nbr]
                adj_lbls[pi, j] = label_to_idx[lbl]
                adj_mask[pi, j] = True

        # Cost matrix: (n_tiles+1, n_orient, n_tiles+1, n_orient, n_labels)
        n_orient = puzzle.n_orientations
        cm = np.zeros((n_tiles + 1, n_orient, n_tiles + 1, n_orient, n_labels))
        for ia, ta in enumerate(tile_set):
            for oa in range(n_orient):
                for ib, tb in enumerate(tile_set):
                    for ob in range(n_orient):
                        for il, l in enumerate(all_labels):
                            cm[ia, oa, ib, ob, il] = puzzle.edge_cost(ta, oa, tb, ob, l)

        # Position cost matrix: (n_tiles+1, n_orient, n_pos)
        pm = np.zeros((n_tiles + 1, n_orient, n_pos))
        if puzzle.position_cost is not None:
            for ia, ta in enumerate(tile_set):
                for oa in range(n_orient):
                    for pi, pos in enumerate(pos_list):
                        pm[ia, oa, pi] = puzzle.position_cost(ta, oa, pos)

        # Constraint matrix: (n_tiles+1, n_orient, n_pos)
        con = np.ones((n_tiles + 1, n_orient, n_pos), dtype=bool)
        if puzzle.constraints is not None:
            for ia, ta in enumerate(tile_set):
                for oa in range(n_orient):
                    for pi, pos in enumerate(pos_list):
                        con[ia, oa, pi] = puzzle.constraints(ta, oa, pos)

        # Initial state
        init_tiles = np.full(n_pos, EMPTY, dtype=np.int32)
        init_orients = np.zeros(n_pos, dtype=np.int32)
        for pos in pos_list:
            pi = pos_to_idx[pos]
            tid = puzzle.tiles[pos]
            if tid is not None:
                init_tiles[pi] = tile_to_idx[tid]
            init_orients[pi] = puzzle.orients[pos]

        # Convert to JAX arrays
        self._cm = jnp.array(cm)
        self._pm = jnp.array(pm)
        self._con = jnp.array(con)
        self._an = jnp.array(adj_nbrs)
        self._al = jnp.array(adj_lbls)
        self._am = jnp.array(adj_mask)
        self._si = swap_idx
        self._init_tiles = jnp.array(init_tiles)
        self._init_orients = jnp.array(init_orients)
        self._n_swap = n_swap
        self._n_orient = n_orient
        self._pos_list = pos_list
        self._tile_to_idx = tile_to_idx
        self._EMPTY = EMPTY

        self._build_fns()

    def _build_fns(self):
        """Create JIT-compiled step, energy, and scan functions."""
        cm, pm, con = self._cm, self._pm, self._con
        an, al, am = self._an, self._al, self._am
        si, ns, no = self._si, self._n_swap, self._n_orient

        def energy_at(tiles, orients, pos):
            tid, ori = tiles[pos], orients[pos]
            pcost = pm[tid, ori, pos]
            ntiles = tiles[an[pos]]
            norients = orients[an[pos]]
            costs = cm[tid, ori, ntiles, norients, al[pos]]
            return pcost + jnp.where(am[pos], costs, 0.0).sum()

        def total_energy(tiles, orients):
            tids = tiles[si]
            ors = orients[si]
            pcosts = pm[tids, ors, si].sum()
            npos = an[si]
            tids_exp = jnp.broadcast_to(tids[:, None], npos.shape)
            ors_exp = jnp.broadcast_to(ors[:, None], npos.shape)
            costs = cm[tids_exp, ors_exp, tiles[npos], orients[npos], al[si]]
            return pcosts + jnp.where(am[si], costs, 0.0).sum()

        def step(carry, temp):
            key, tiles, orients = carry
            key, k1, k2, k3, k4, k5, k6 = jax.random.split(key, 7)

            # --- Swap move ---
            idxs = jax.random.randint(k1, (2,), 0, ns)
            idxs = idxs.at[1].set(
                jnp.where(idxs[0] == idxs[1], (idxs[1] + 1) % ns, idxs[1])
            )
            p1, p2 = si[idxs[0]], si[idxs[1]]

            s_old = energy_at(tiles, orients, p1) + energy_at(tiles, orients, p2)
            s_tiles = tiles.at[p1].set(tiles[p2]).at[p2].set(tiles[p1])
            s_orients = orients.at[p1].set(orients[p2]).at[p2].set(orients[p1])
            s_new = energy_at(s_tiles, s_orients, p1) + energy_at(s_tiles, s_orients, p2)

            s_delta = s_new - s_old
            s_accept = (s_delta <= 0) | (
                jax.random.uniform(k2) < jnp.exp(jnp.clip(-s_delta / temp, -20.0, 0.0))
            )
            s_accept = s_accept & con[tiles[p1], orients[p1], p2] & con[tiles[p2], orients[p2], p1]

            # --- Rotate move ---
            r_pos = si[jax.random.randint(k3, (), 0, ns)]
            r_ori = jax.random.randint(k4, (), 0, jnp.maximum(no, 1))

            r_old = energy_at(tiles, orients, r_pos)
            r_orients = orients.at[r_pos].set(r_ori)
            r_new = energy_at(tiles, r_orients, r_pos)

            r_delta = r_new - r_old
            r_accept = (r_delta <= 0) | (
                jax.random.uniform(k5) < jnp.exp(jnp.clip(-r_delta / temp, -20.0, 0.0))
            )
            r_accept = r_accept & con[tiles[r_pos], r_ori, r_pos]

            # --- Choose move type ---
            do_rotate = (no > 1) & (jax.random.uniform(k6) < 0.3)

            final_tiles = jnp.where(do_rotate, tiles, jnp.where(s_accept, s_tiles, tiles))
            final_orients = jnp.where(
                do_rotate,
                jnp.where(r_accept, r_orients, orients),
                jnp.where(s_accept, s_orients, orients),
            )

            return (key, final_tiles, final_orients), None

        self._step_fn = step
        self._total_energy_fn = jax.jit(total_energy)

        @jax.jit
        def run_chunk(carry, temps):
            return lax.scan(step, carry, temps)

        self._run_chunk = run_chunk

    def run(self, start_temp, end_temp, num_steps, log_rate=10_000, seed=0):
        """Run a single SA chain with periodic logging."""
        num_steps = int(num_steps)
        log_rate = int(log_rate)
        temps = jnp.logspace(jnp.log10(start_temp), jnp.log10(end_temp), num_steps)
        key = jax.random.key(seed)
        tiles, orients = self._init_tiles, self._init_orients

        t0 = time.perf_counter()
        for start in range(0, num_steps, log_rate):
            end = min(start + log_rate, num_steps)
            (key, tiles, orients), _ = self._run_chunk(
                (key, tiles, orients), temps[start:end]
            )
            tiles.block_until_ready()
            e = float(self._total_energy_fn(tiles, orients))
            elapsed = time.perf_counter() - t0
            print(f"step {end}/{num_steps}  E={e:.1f}  T={float(temps[end - 1]):.2f}  t={elapsed:.1f}s")

        self.energy = float(self._total_energy_fn(tiles, orients))
        self._write_back(tiles, orients)

    def run_parallel(self, start_temp, end_temp, num_steps, n_chains=64, seed=0):
        """Run multiple SA chains in parallel, keep the best."""
        num_steps = int(num_steps)
        temps = jnp.logspace(jnp.log10(start_temp), jnp.log10(end_temp), num_steps)
        keys = jax.random.split(jax.random.key(seed), n_chains)
        init_tiles = jnp.tile(self._init_tiles, (n_chains, 1))
        init_orients = jnp.tile(self._init_orients, (n_chains, 1))

        step_fn = self._step_fn

        @jax.jit
        def run_all(keys, tiles, orients):
            def one_chain(key, t, o):
                (_, ft, fo), _ = lax.scan(step_fn, (key, t, o), temps)
                return ft, fo
            return jax.vmap(one_chain)(keys, tiles, orients)

        t0 = time.perf_counter()
        all_tiles, all_orients = run_all(keys, init_tiles, init_orients)
        all_tiles.block_until_ready()
        elapsed = time.perf_counter() - t0

        energies = jax.vmap(self._total_energy_fn)(all_tiles, all_orients)
        best = int(jnp.argmin(energies))
        self.energy = float(energies[best])

        print(f"{n_chains} chains x {num_steps} steps in {elapsed:.1f}s  best E={self.energy:.1f}")
        self._write_back(all_tiles[best], all_orients[best])

    def _write_back(self, tiles, orients):
        """Copy JAX results back into the Puzzle object."""
        tiles_np = np.array(tiles)
        orients_np = np.array(orients)
        idx_to_tile = {i: t for t, i in self._tile_to_idx.items()}
        for pi, pos in enumerate(self._pos_list):
            tidx = int(tiles_np[pi])
            self.puzzle.tiles[pos] = idx_to_tile.get(tidx)
            self.puzzle.orients[pos] = int(orients_np[pi])
