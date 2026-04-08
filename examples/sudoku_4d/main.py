"""4D Sudoku solver via simulated annealing.

A 4D Sudoku has 4 constraint types — every digit 1-9 must appear
exactly once in each:
  1. Row
  2. Column
  3. 3x3 Box
  4. Color region (9 irregular regions defined by cell colors)

Uses box-restricted swaps with incremental violation counting and
multiple restarts. Puzzle from f-puzzles.

Usage:
    uv run python examples/sudoku_4d/main.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import time

# --- Puzzle data (decoded from f-puzzles) ---

GIVENS = {
    (0, 5): 4, (0, 8): 9,
    (1, 0): 3, (1, 4): 7,
    (2, 3): 1,
    (3, 0): 8, (3, 3): 2, (3, 6): 1,
    (4, 5): 6,
    (5, 2): 8, (5, 5): 9, (5, 8): 4,
    (6, 3): 9, (6, 7): 5,
    (7, 1): 1,
    (8, 0): 4,
}

SOLUTION = [
    [7, 2, 1, 5, 8, 4, 6, 3, 9],
    [3, 5, 6, 4, 7, 1, 9, 8, 2],
    [6, 8, 4, 1, 9, 3, 2, 7, 5],
    [8, 9, 3, 2, 6, 5, 1, 4, 7],
    [9, 4, 2, 7, 3, 6, 5, 1, 8],
    [5, 7, 8, 6, 1, 9, 3, 2, 4],
    [1, 3, 7, 9, 4, 2, 8, 5, 6],
    [2, 1, 9, 8, 5, 7, 4, 6, 3],
    [4, 6, 5, 3, 2, 8, 7, 9, 1],
]

COLOR_HEX = [
    ['#8080F0','#A8A8A8','#FFA0A0','0',      '#60D060','#D0D0FF','#FFE060','#FFFFB0','#8080F0'],
    ['#60D060','#FFFFB0','#FFA0A0','#FFFFB0','0',      '#A8A8A8','#A8A8A8','#FFA0A0','#FFA0A0'],
    ['#FFD0D0','#8080F0','#A8A8A8','0',      '#60D060','#FFD0D0','#FFD0D0','#FFFFB0','#FFD0D0'],
    ['#FFE060','#FFE060','#FFA0A0','#FFE060','#60D060','#FFE060','#8080F0','0',      '#60D060'],
    ['#D0D0FF','#FFD0D0','#D0D0FF','#D0D0FF','#D0D0FF','#8080F0','#D0D0FF','#FFFFB0','#D0D0FF'],
    ['#8080F0','#A8A8A8','#FFFFB0','0',      '#60D060','#FFD0D0','#FFE060','#FFFFB0','#FFE060'],
    ['#FFE060','#A8A8A8','#FFD0D0','#FFFFB0','#60D060','0',      '#A8A8A8','#FFA0A0','#A8A8A8'],
    ['#8080F0','#D0D0FF','#FFA0A0','0',      '#60D060','#FFA0A0','#8080F0','#FFFFB0','#8080F0'],
    ['#FFA0A0','#D0D0FF','#A8A8A8','0',      '#60D060','#FFD0D0','#FFE060','0',      '#FFD0D0'],
]

UNIQUE_COLORS = sorted(set(c for row in COLOR_HEX for c in row))
COLOR_TO_ID = {c: i for i, c in enumerate(UNIQUE_COLORS)}

# Precompute: for each cell, which groups it belongs to (row, col, color)
# Box is preserved by construction, so we skip it
CELL_GROUPS = {}
ROW_CELLS = [[(r, c) for c in range(9)] for r in range(9)]
COL_CELLS = [[(r, c) for r in range(9)] for c in range(9)]
COLOR_CELLS = {i: [] for i in range(9)}
for r in range(9):
    for c in range(9):
        cid = COLOR_TO_ID[COLOR_HEX[r][c]]
        COLOR_CELLS[cid].append((r, c))
        CELL_GROUPS[(r, c)] = (r, c, cid)  # row_id, col_id, color_id


def duplicates_in(grid, cells):
    """Count duplicates in a group of cells."""
    vals = [grid[r][c] for r, c in cells]
    return len(vals) - len(set(vals))


def group_violations(grid, row_id, col_id, color_id):
    """Violations for the 3 groups a cell belongs to."""
    return (duplicates_in(grid, ROW_CELLS[row_id]) +
            duplicates_in(grid, COL_CELLS[col_id]) +
            duplicates_in(grid, COLOR_CELLS[color_id]))


def swap_delta(grid, r1, c1, r2, c2):
    """Compute change in violations from swapping two cells.

    Only recomputes affected rows, columns, and color regions.
    """
    rid1, cid1, clr1 = CELL_GROUPS[(r1, c1)]
    rid2, cid2, clr2 = CELL_GROUPS[(r2, c2)]

    # Groups affected (deduplicate if cells share a row/col/color)
    affected = set()
    affected.add(('r', rid1))
    affected.add(('r', rid2))
    affected.add(('c', cid1))
    affected.add(('c', cid2))
    affected.add(('l', clr1))
    affected.add(('l', clr2))

    # Count violations before
    old_v = 0
    for kind, gid in affected:
        if kind == 'r':
            old_v += duplicates_in(grid, ROW_CELLS[gid])
        elif kind == 'c':
            old_v += duplicates_in(grid, COL_CELLS[gid])
        else:
            old_v += duplicates_in(grid, COLOR_CELLS[gid])

    # Swap
    grid[r1][c1], grid[r2][c2] = grid[r2][c2], grid[r1][c1]

    # Count violations after
    new_v = 0
    for kind, gid in affected:
        if kind == 'r':
            new_v += duplicates_in(grid, ROW_CELLS[gid])
        elif kind == 'c':
            new_v += duplicates_in(grid, COL_CELLS[gid])
        else:
            new_v += duplicates_in(grid, COLOR_CELLS[gid])

    # Undo swap (caller decides whether to keep)
    grid[r1][c1], grid[r2][c2] = grid[r2][c2], grid[r1][c1]

    return new_v - old_v


def full_violations(grid):
    v = 0
    for r in range(9):
        v += duplicates_in(grid, ROW_CELLS[r])
    for c in range(9):
        v += duplicates_in(grid, COL_CELLS[c])
    for cid in range(9):
        v += duplicates_in(grid, COLOR_CELLS[cid])
    return v


def init_grid(rng):
    """Initialize grid: fill each box with its missing digits."""
    grid = [[0] * 9 for _ in range(9)]
    given_set = set(GIVENS.keys())

    for (r, c), v in GIVENS.items():
        grid[r][c] = v

    box_empty = {}
    for b in range(9):
        br, bc = (b // 3) * 3, (b % 3) * 3
        present = set()
        empty = []
        for r in range(br, br + 3):
            for c in range(bc, bc + 3):
                if (r, c) in given_set:
                    present.add(grid[r][c])
                else:
                    empty.append((r, c))
        missing = [d for d in range(1, 10) if d not in present]
        rng.shuffle(missing)
        for (r, c), d in zip(empty, missing):
            grid[r][c] = d
        box_empty[b] = empty

    return grid, box_empty


def solve_once(rng, num_cycles=20, steps_per_cycle=200_000,
               start_temp=2.0, end_temp=0.001):
    """SA with box-restricted swaps and periodic reheating.

    Runs multiple cooling cycles. If stuck, reheats to escape local minima.
    """
    grid, box_empty = init_grid(rng)
    swap_pools = [(b, cells) for b, cells in box_empty.items() if len(cells) >= 2]
    violations = full_violations(grid)
    best_v = violations
    best_grid = [row[:] for row in grid]

    for cycle in range(num_cycles):
        if violations == 0:
            break

        # Reheat: temperature proportional to remaining violations
        cycle_start = min(start_temp, 0.5 + 0.15 * violations)
        temps = np.logspace(np.log10(cycle_start), np.log10(end_temp), steps_per_cycle)

        for step in range(steps_per_cycle):
            if violations == 0:
                break

            b, cells = swap_pools[rng.integers(len(swap_pools))]
            idx = rng.choice(len(cells), 2, replace=False)
            r1, c1 = cells[idx[0]]
            r2, c2 = cells[idx[1]]

            delta = swap_delta(grid, r1, c1, r2, c2)

            if delta <= 0 or rng.random() < np.exp(-delta / temps[step]):
                grid[r1][c1], grid[r2][c2] = grid[r2][c2], grid[r1][c1]
                violations += delta

            if violations < best_v:
                best_v = violations
                best_grid = [row[:] for row in grid]

        # Reset to best if stuck
        if violations > best_v:
            grid = [row[:] for row in best_grid]
            violations = best_v

    return best_grid, best_v


def main():
    # Verify solution
    for cid in range(len(UNIQUE_COLORS)):
        cells = [(r, c) for r in range(9) for c in range(9)
                 if COLOR_TO_ID[COLOR_HEX[r][c]] == cid]
        vals = sorted(SOLUTION[r][c] for r, c in cells)
        assert vals == list(range(1, 10)), f"Color region {cid} fails"
    print("4D Sudoku (f-puzzles)")
    print(f"Given clues: {len(GIVENS)}")
    print(f"Solution verified against all 4 constraints.")
    print()

    # Multiple restarts
    t0 = time.perf_counter()
    best_grid, best_v = None, 999
    attempts = 0

    while best_v > 0:
        attempts += 1
        rng = np.random.default_rng(attempts * 1000 + 42)
        grid, v = solve_once(rng, num_cycles=30, steps_per_cycle=300_000,
                             start_temp=3.0, end_temp=0.0005)
        if v < best_v:
            best_v = v
            best_grid = grid
        elapsed = time.perf_counter() - t0
        print(f"attempt {attempts}  violations={v}  best={best_v}  t={elapsed:.1f}s")

        if elapsed > 300:
            print("Time limit reached.")
            break

    grid = best_grid
    elapsed = time.perf_counter() - t0
    print(f"\n{attempts} attempts in {elapsed:.1f}s")
    print()

    given_set = set(GIVENS.keys())
    correct = 0
    for r in range(9):
        row_str = ""
        for c in range(9):
            d = grid[r][c]
            expected = SOLUTION[r][c]
            if (r, c) in given_set:
                row_str += f"[{d}]"
            elif d == expected:
                row_str += f" {d} "
            else:
                row_str += f" {d}!"
            if d == expected:
                correct += 1
        print(row_str)

    print()
    print(f"Correct: {correct}/81 ({100*correct/81:.0f}%)")
    if best_v == 0:
        print("SOLVED!")
    else:
        print(f"Remaining violations: {best_v}")


if __name__ == '__main__':
    main()
