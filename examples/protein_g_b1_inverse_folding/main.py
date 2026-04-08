"""Protein sequence design via simulated annealing.

Given a real protein's 3D structure (PDB file), compute the residue
contact graph from C-alpha distances, then find the amino acid sequence
that minimizes total pairwise contact energy. This is the inverse
protein folding problem.

Uses the B1 domain of Streptococcal Protein G (PDB: 1PGA), a classic
56-residue protein with a 4-stranded beta-sheet + alpha-helix fold.

Unlike the HP lattice model (2 amino acid types), this uses all 20
amino acid types with a pairwise energy matrix derived from amino acid
physicochemical properties.

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

THREE_TO_ONE = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
    'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
    'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
}

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

    Three terms:
      - Hydrophobic packing: favorable only when both residues are
        hydrophobic (positive Kyte-Doolittle). Burying hydrophilic
        residues is penalized.
      - Electrostatics: opposite charges attract (salt bridges),
        like charges repel.
      - Disulfide: Cys-Cys contacts get a strong bonus.
    """
    n = len(AMINO_ACIDS)
    E = np.zeros((n, n))
    for i, aa1 in enumerate(AMINO_ACIDS):
        for j, aa2 in enumerate(AMINO_ACIDS):
            h1, h2 = HYDROPHOBICITY[aa1], HYDROPHOBICITY[aa2]
            c1, c2 = CHARGE[aa1], CHARGE[aa2]
            # Hydrophobic packing: reward burying two hydrophobic residues,
            # penalize burying hydrophilic ones
            E[i, j] = -0.1 * max(h1, 0) * max(h2, 0)  # favorable for hydrophobic pairs
            E[i, j] += 0.05 * max(-h1, 0) * max(-h2, 0)  # penalty for hydrophilic pairs buried
            E[i, j] += 0.03 * abs(min(h1, 0) * max(h2, 0) + max(h1, 0) * min(h2, 0))  # mixed penalty
            # Electrostatics: opposite charges attract
            E[i, j] -= 0.5 * c1 * c2
            # Disulfide
            if aa1 == 'C' and aa2 == 'C':
                E[i, j] -= 2.0
    return E


# --- PDB parsing ---

def parse_pdb(path):
    """Extract C-alpha coordinates and sequence from a PDB file.

    Returns:
        coords: np.array of shape (n_residues, 3)
        sequence: list of one-letter amino acid codes
    """
    coords = []
    sequence = []
    seen = set()
    with open(path) as f:
        for line in f:
            if not line.startswith('ATOM'):
                continue
            atom_name = line[12:16].strip()
            if atom_name != 'CA':
                continue
            res_name = line[17:20].strip()
            chain = line[21]
            res_seq = int(line[22:26])
            key = (chain, res_seq)
            if key in seen:
                continue
            seen.add(key)
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            coords.append([x, y, z])
            aa = THREE_TO_ONE.get(res_name, 'A')
            sequence.append(aa)
    return np.array(coords), sequence


def build_contact_graph(coords, cutoff=8.0, min_seq_sep=3):
    """Build a contact graph from C-alpha coordinates.

    Two residues are in contact if their C-alpha distance is below the
    cutoff and they are at least min_seq_sep apart in sequence (to
    exclude trivial backbone neighbors).
    """
    n = len(coords)
    adjacency = {i: [] for i in range(n)}

    for i in range(n):
        for j in range(i + min_seq_sep, n):
            dist = np.linalg.norm(coords[i] - coords[j])
            if dist < cutoff:
                adjacency[i].append((j, 'contact'))
                adjacency[j].append((i, 'contact'))

    return adjacency


def main():
    here = os.path.dirname(__file__)
    pdb_path = os.path.join(here, '1pga.pdb')

    if not os.path.exists(pdb_path):
        print("Downloading PDB 1PGA (Protein G B1 domain)...")
        import urllib.request
        urllib.request.urlretrieve(
            'https://files.rcsb.org/download/1PGA.pdb', pdb_path
        )

    # Parse structure
    coords, native_seq = parse_pdb(pdb_path)
    n_residues = len(native_seq)
    print(f"Protein G B1 domain (PDB: 1PGA)")
    print(f"Residues: {n_residues}")
    print(f"Native:   {''.join(native_seq)}")

    # Build contact graph from real 3D coordinates
    adjacency = build_contact_graph(coords, cutoff=8.0, min_seq_sep=3)
    n_contacts = sum(len(v) for v in adjacency.values()) // 2
    print(f"Contacts: {n_contacts} (8A cutoff, |i-j|>=3)")

    # Energy matrix
    energy_matrix = build_energy_matrix()

    def edge_cost(aa_a, orient_a, aa_b, orient_b, label):
        return energy_matrix[aa_a, aa_b]

    # Native energy
    native_ids = [AA_TO_IDX[aa] for aa in native_seq]
    native_puzzle = Puzzle(adjacency, native_ids, edge_cost)
    native_energy = native_puzzle.total_energy()
    print(f"Native energy: {native_energy:.1f}")

    # Shuffle and solve
    shuffled_ids = native_ids.copy()
    np.random.default_rng(123).shuffle(shuffled_ids)
    puzzle = Puzzle(adjacency, shuffled_ids, edge_cost)
    print(f"Shuffled energy: {puzzle.total_energy():.1f}")
    print()

    solver = JaxAnneal(puzzle)
    solver.run(start_temp=50, end_temp=0.01, num_steps=10e6,
               log_rate=500_000, sigma=25)

    # Report
    designed_seq = [AMINO_ACIDS[puzzle.tiles[i]] for i in range(n_residues)]
    matches = sum(a == b for a, b in zip(native_seq, designed_seq))
    print()
    print(f"Native:   {''.join(native_seq)}")
    print(f"Designed: {''.join(designed_seq)}")
    print(f"Sequence identity: {matches}/{n_residues} ({100*matches/n_residues:.0f}%)")
    print(f"Native energy:   {native_energy:.1f}")
    print(f"Designed energy: {solver.energy:.1f}")

    if solver.energy <= native_energy:
        print("Designed sequence matches or beats native energy.")
    else:
        print(f"Gap to native: {solver.energy - native_energy:.1f}")

    # Show burial analysis
    print()
    print("Burial analysis (contacts -> residue type):")
    n_cont = {i: len(nbrs) for i, nbrs in adjacency.items()}
    for label, lo, hi in [("buried (>=8)", 8, 99), ("semi-buried (5-7)", 5, 7), ("exposed (<5)", 0, 4)]:
        positions = [i for i in range(n_residues) if lo <= n_cont[i] <= hi]
        native_aas = ''.join(native_seq[i] for i in positions)
        designed_aas = ''.join(designed_seq[i] for i in positions)
        print(f"  {label:20s}  native={native_aas}  designed={designed_aas}")


if __name__ == '__main__':
    main()
