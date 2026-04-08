# sim-anneal

N-dimensional simulated annealing puzzle solver. Provide tiles and an edge cost function, get an assembled solution.

## Algorithm

* https://en.wikipedia.org/wiki/Simulated_annealing

## Sampling

* https://en.wikipedia.org/wiki/Metropolis%E2%80%93Hastings_algorithm

## Usage

Supply an `edge_cost(tile_a, tile_b, axis) -> float` function and a grid shape:

```python
from puzzle import Puzzle, Anneal

puzzle = Puzzle(shape=(10, 10), tile_ids=ids, edge_cost=my_cost)
Anneal(puzzle).run(start_temp=100, end_temp=1, num_steps=1e6)
```

Works for any number of dimensions (1D, 2D, 3D, ...).

## Image example

`main.py` solves a 2D image jigsaw using pixel RMSE at tile borders as the edge cost.

```
uv run python main.py
```

## Tests

```
uv run pytest
```
