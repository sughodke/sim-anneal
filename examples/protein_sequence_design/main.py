"""Protein sequence design via simulated annealing.

Given a protein's 3D contact graph, find the amino acid sequence that
minimizes total pairwise contact energy. This is the inverse protein
folding problem.

Unlike the HP lattice model (2 amino acid types), this uses all 20
amino acid types with a pairwise energy matrix derived from amino acid
physicochemical properties (hydrophobicity, charge, disulfide bonding).

The protein is a 30-residue helix-loop-strand motif with:
  - alpha helix contacts (i, i+3) and (i, i+4)
  - anti-parallel beta sheet contacts between helix and strand
  - a buried hydrophobic core and solvent-exposed charged residues

Usage:
    uv run python examples/protein_sequence_design/main.py
"""

import sys, os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
from puzzle import Puzzle
from jax_anneal import JaxAnneal

# --- Amino acid definitions ---

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
AA_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}

# Kyte-Doolittle hydrophobicity scale
HYDROPHOBICITY = {
    'I': 4.5, 'V': 4.2, 'L': 3.8, 'F': 2.8, 'C': 2.5,
    'M': 1.9, 'A': 1.8, 'G': -0.4, 'T': -0.7, 'S': -0.8,
    'W': -0.9, 'Y': -1.3, 'P': -1.6, 'H': -3.2, 'D': -3.5,
    'E': -3.5, 'N': -3.5, 'Q': -3.5, 'K': -3.9, 'R': -4.5,
}

# Formal charge at pH 7
CHARGE = {aa: 0.0 for aa in AMINO_ACIDS}
CHARGE.update({'D': -1.0, 'E': -1.0, 'K': 1.0, 'R': 1.0, 'H': 0.5})


def build_energy_matrix():
    """Build a 20x20 contact energy matrix from physicochemical properties.

    Captures three key interactions:
      - Hydrophobic effect: burying nonpolar residues together is favorable
      - Electrostatics: opposite charges attract, like charges repel
      - Disulfide bonding: Cys-Cys contacts are strongly favorable
    """
    n = len(AMINO_ACIDS)
    E = np.zeros((n, n))
    for i, aa1 in enumerate(AMINO_ACIDS):
        for j, aa2 in enumerate(AMINO_ACIDS):
            h1, h2 = HYDROPHOBICITY[aa1], HYDROPHOBICITY[aa2]
            c1, c2 = CHARGE[aa1], CHARGE[aa2]
            # Hydrophobic: favorable when both are hydrophobic (positive h)
            E[i, j] = -0.1 * h1 * h2
            # Electrostatic
            E[i, j] += 0.5 * c1 * c2
            # Disulfide
            if aa1 == 'C' and aa2 == 'C':
                E[i, j] -= 2.0
    return E


def build_contact_graph(n_residues=30):
    """Build a contact graph for a helix-loop-strand protein.

    Residues 0-12:  alpha helix
    Residues 13-16: loop
    Residues 17-29: beta strand (anti-parallel to helix)

    Contacts:
      - Sequential backbone: (i, i+1) for all i
      - Helix: (i, i+3) and (i, i+4) within helix
      - Beta sheet: anti-parallel pairing between strand and helix
    """
    contacts = set()

    # Sequential backbone bonds
    for i in range(n_residues - 1):
        contacts.add((i, i + 1))

    # Alpha helix contacts (i, i+3) and (i, i+4)
    helix_end = 12
    for i in range(helix_end + 1):
        if i + 3 <= helix_end:
            contacts.add((i, i + 3))
        if i + 4 <= helix_end:
            contacts.add((i, i + 4))

    # Anti-parallel beta sheet: strand residues contact helix residues
    # Strand 17-29 pairs with helix 0-12 in anti-parallel fashion
    sheet_pairs = [
        (17, 12), (18, 11), (19, 10), (20, 9), (21, 8),
        (22, 7), (23, 6), (24, 5), (25, 4), (26, 3),
    ]
    for a, b in sheet_pairs:
        contacts.add((a, b))

    # Build adjacency dict
    adjacency = {i: [] for i in range(n_residues)}
    for a, b in contacts:
        adjacency[a].append((b, 'contact'))
        adjacency[b].append((a, 'contact'))

    return adjacency


def design_native_sequence(adjacency, n_residues=30):
    """Design a 'native' sequence with favorable energetics.

    Buried positions (many contacts) get hydrophobic residues.
    Exposed positions (few contacts) get polar/charged residues.
    Add a Cys-Cys pair at contacting positions for a disulfide bond.
    """
    # Count contacts per position (proxy for burial)
    n_contacts = {i: len(nbrs) for i, nbrs in adjacency.items()}

    hydrophobic = list("IVLFM")
    polar = list("STNGQ")
    charged = list("DEKRH")

    sequence = ['A'] * n_residues
    rng = np.random.default_rng(42)

    for i in range(n_residues):
        nc = n_contacts[i]
        if nc >= 5:  # buried — hydrophobic
            sequence[i] = rng.choice(hydrophobic)
        elif nc >= 3:  # semi-buried — polar
            sequence[i] = rng.choice(polar)
        else:  # exposed — charged/polar
            sequence[i] = rng.choice(charged + polar)

    # Add a disulfide bond at a contacting pair
    for a, b in [(20, 9)]:  # a sheet-helix contact pair
        sequence[a] = 'C'
        sequence[b] = 'C'

    return sequence


def main():
    n_residues = 30
    adjacency = build_contact_graph(n_residues)
    native_seq = design_native_sequence(adjacency, n_residues)

    print(f"Native:   {''.join(native_seq)}")
    print(f"Residues: {n_residues}")
    print(f"Contacts: {sum(len(v) for v in adjacency.values()) // 2}")

    # Build energy matrix
    energy_matrix = build_energy_matrix()

    def edge_cost(aa_a, orient_a, aa_b, orient_b, label):
        return energy_matrix[aa_a, aa_b]

    # Compute native energy
    native_ids = [AA_TO_IDX[aa] for aa in native_seq]
    native_puzzle = Puzzle(adjacency, native_ids, edge_cost)
    native_energy = native_puzzle.total_energy()
    print(f"Native energy: {native_energy:.1f}")

    # Shuffle the sequence
    shuffled_ids = native_ids.copy()
    np.random.default_rng(123).shuffle(shuffled_ids)
    puzzle = Puzzle(adjacency, shuffled_ids, edge_cost)
    shuffled_energy = puzzle.total_energy()
    print(f"Shuffled energy: {shuffled_energy:.1f}")
    print()

    # Solve
    solver = JaxAnneal(puzzle)
    solver.run(start_temp=50, end_temp=0.01, num_steps=2e6, log_rate=100_000, sigma=15)

    # Report
    designed_seq = [AMINO_ACIDS[puzzle.tiles[i]] for i in range(n_residues)]
    print()
    print(f"Native:   {''.join(native_seq)}")
    print(f"Designed: {''.join(designed_seq)}")

    matches = sum(a == b for a, b in zip(native_seq, designed_seq))
    print(f"Sequence identity: {matches}/{n_residues} ({100*matches/n_residues:.0f}%)")
    print(f"Native energy:   {native_energy:.1f}")
    print(f"Designed energy: {solver.energy:.1f}")

    if solver.energy <= native_energy:
        print("Designed sequence matches or beats native energy.")
    else:
        print(f"Gap to native: {solver.energy - native_energy:.1f}")


if __name__ == '__main__':
    main()
