# sim-anneal

N-dimensional simulated annealing puzzle solver. Provide tiles, an adjacency graph, and an edge cost function — get an assembled solution.

## Algorithm

* https://en.wikipedia.org/wiki/Simulated_annealing

## Sampling

* https://en.wikipedia.org/wiki/Metropolis%E2%80%93Hastings_algorithm

## Usage

```python
from puzzle import Puzzle
from jax_anneal import JaxAnneal

def edge_cost(a, orient_a, b, orient_b, label):
    return ...  # cost of placing tile a next to tile b

puzzle = Puzzle.grid(shape=(10, 10), tile_ids=ids, edge_cost=edge_cost)
solver = JaxAnneal(puzzle)
solver.run_parallel(start_temp=100, end_temp=1, num_steps=1e6, n_chains=16, sigma=30)
```

Supports n-dimensional grids, chains, arbitrary graphs, tile orientations, position-dependent energy, hard constraints, and partial occupancy.

## Examples

| Example | Domain | Size | Time | Result |
|---------|--------|------|------|--------|
| `bremen_town_musicians_jigsaw` | 2D image jigsaw | 216 tiles | ~45s | E≈20K, image clearly recognizable |
| `protein_g_b1_inverse_folding` | Inverse protein folding (real PDB structure) | 56 residues, 137 contacts | ~41s | Hydrophobic core packed correctly |
| `plusplus_voxel_cat` | 3D voxel reassembly + PLY export | 536 voxels in 14×10×16 grid | ~90s | E≈15K, color regions in right places |
| `sudoku_4d` | 4D Sudoku (row + col + box + color region) | 9×9, 65 cells, 4 constraints | ~5min | ~12 violations (near-solved) |

Run any example from the repo root:

```
uv run python examples/bremen_town_musicians_jigsaw/main.py
```

## Tests

```
uv run pytest
```
