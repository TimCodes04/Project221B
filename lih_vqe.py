"""LiH dissociation curve via VQE with a UCCSD ansatz.

Pipeline (matches the project proposal):

    1. PySCF: RHF + FCI + one- and two-electron MO integrals at each Li-H
       distance.
    2. Freeze the Li 1s core analytically (effective h_pq + inactive energy)
       to keep the qubit count manageable.
    3. Build the active-space second-quantized Hamiltonian directly from the
       integrals using the Slater-Condon rules.
    4. Map the FermionicOp to qubits with the Jordan-Wigner transformation.
    5. UCCSD ansatz on top of the Hartree-Fock reference, optimized with
       COBYLA on the Qiskit Aer statevector simulator.
    6. Bootstrap optimal parameters to the next geometry to accelerate the
       sweep across bond lengths.

Outputs: dissociation_curve.png and correlation_energy.png.
"""

from __future__ import annotations

import itertools
import time

import matplotlib.pyplot as plt
import numpy as np
from pyscf import ao2mo, fci, gto, scf

from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import EstimatorV2 as AerEstimator
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import COBYLA
from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.operators import FermionicOp


N_FROZEN_CORE = 1  # freeze the Li 1s spatial orbital
INTEGRAL_TOL = 1e-12


# ---------------------------------------------------------------------------
# 1. Classical pre-processing (PySCF)
# ---------------------------------------------------------------------------
def run_pyscf(distance: float) -> dict:
    """RHF + FCI + MO integrals for LiH at the given Li-H distance (Angstrom)."""
    mol = gto.M(
        atom=f"Li 0 0 0; H 0 0 {distance}",
        basis="sto-3g",
        unit="Angstrom",
        verbose=0,
    )
    mf = scf.RHF(mol)
    mf.kernel()

    n_orb = mf.mo_coeff.shape[1]
    h1_ao = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
    h1_mo = mf.mo_coeff.T @ h1_ao @ mf.mo_coeff
    eri_mo = ao2mo.kernel(mol, mf.mo_coeff, compact=False).reshape(
        n_orb, n_orb, n_orb, n_orb
    )

    e_fci, _ = fci.FCI(mf).kernel()

    return {
        "e_hf": float(mf.e_tot),
        "e_fci": float(e_fci),
        "e_nuc": float(mol.energy_nuc()),
        "h1_mo": h1_mo,
        "eri_mo": eri_mo,  # chemist's notation: eri_mo[p, q, r, s] = (pq|rs)
        "n_orb": n_orb,
        "n_elec": int(mol.nelectron),
    }


# ---------------------------------------------------------------------------
# 2. Frozen-core projection
# ---------------------------------------------------------------------------
def freeze_core(
    h1: np.ndarray,
    eri: np.ndarray,
    e_nuc: float,
    n_orb: int,
    n_frozen: int,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    """Project closed-shell core orbitals into an inactive energy + effective h1.

    With orbitals 0 .. n_frozen-1 doubly occupied and frozen at the HF values,
    the active-space Hamiltonian is

        H = E_inactive
            + sum_{pq in active}  h_eff[p,q] * sum_sigma a^dag_{p,sigma} a_{q,sigma}
            + 1/2 sum_{pqrs in active} (pq|rs) * sum_{sigma,tau} a^dag a^dag a a

    where (i,j run over inactive / frozen orbitals)

        E_inactive   = E_nuc + 2 sum_i h[i,i] + sum_ij [2 (ii|jj) - (ij|ji)]
        h_eff[p, q]  = h[p, q] + sum_i [2 (pq|ii) - (pi|iq)]
    """
    n_active = n_orb - n_frozen

    e_inactive = e_nuc
    for i in range(n_frozen):
        e_inactive += 2.0 * h1[i, i]
    for i, j in itertools.product(range(n_frozen), repeat=2):
        e_inactive += 2.0 * eri[i, i, j, j] - eri[i, j, j, i]

    h1_eff = h1[n_frozen:, n_frozen:].copy()
    for p, q in itertools.product(range(n_active), repeat=2):
        P, Q = p + n_frozen, q + n_frozen
        for i in range(n_frozen):
            h1_eff[p, q] += 2.0 * eri[P, Q, i, i] - eri[P, i, i, Q]

    eri_act = eri[n_frozen:, n_frozen:, n_frozen:, n_frozen:].copy()
    return h1_eff, eri_act, float(e_inactive), n_active


# ---------------------------------------------------------------------------
# 3. Slater-Condon -> FermionicOp
# ---------------------------------------------------------------------------
def build_fermionic_hamiltonian(
    h1: np.ndarray, eri: np.ndarray, n_orb: int
) -> FermionicOp:
    """Construct the spin-resolved second-quantized Hamiltonian.

    Spin-orbital ordering: alpha occupies indices [0, n_orb), beta occupies
    [n_orb, 2 n_orb). The two-electron integrals are in chemist's notation
    (pq|rs); the chemist <-> physicist conversion fixes the operator order

        H_2 = 1/2 sum_{pqrs} sum_{sigma,tau} (pq|rs)
                  a^dag_{p,sigma} a^dag_{r,tau} a_{s,tau} a_{q,sigma}

    Same-spin terms with p == r or q == s vanish by Pauli exclusion and are
    skipped to keep the operator dictionary clean.
    """
    op: dict[str, float] = {}

    # One-electron part: sum_pq h_pq sum_sigma a^dag_{p,sigma} a_{q,sigma}
    for p, q in itertools.product(range(n_orb), repeat=2):
        v = float(h1[p, q])
        if abs(v) < INTEGRAL_TOL:
            continue
        for off in (0, n_orb):  # sigma = alpha (0) and beta (n_orb)
            key = f"+_{p + off} -_{q + off}"
            op[key] = op.get(key, 0.0) + v

    # Two-electron part
    for p, q, r, s in itertools.product(range(n_orb), repeat=4):
        v = float(eri[p, q, r, s])
        if abs(v) < INTEGRAL_TOL:
            continue
        half = 0.5 * v
        for s_off, t_off in itertools.product((0, n_orb), repeat=2):
            ip, iq = p + s_off, q + s_off
            ir, is_ = r + t_off, s + t_off
            if ip == ir or iq == is_:
                continue  # vanishes by anticommutation
            key = f"+_{ip} +_{ir} -_{is_} -_{iq}"
            op[key] = op.get(key, 0.0) + half

    return FermionicOp(op, num_spin_orbitals=2 * n_orb)


# ---------------------------------------------------------------------------
# 4-5. VQE driver
# ---------------------------------------------------------------------------
def make_ansatz(n_active_orb: int, n_particles: tuple[int, int], mapper, pass_mgr):
    """UCCSD on top of HF, transpiled into Aer's native basis.

    The transpile step expands each PauliEvolutionGate into elementary gates
    so the AerSimulator backend (which doesn't understand the EvolvedOps
    instruction emitted by qiskit_nature's UCCSD) can execute the circuit.
    """
    hf = HartreeFock(n_active_orb, n_particles, mapper)
    raw = UCCSD(n_active_orb, n_particles, mapper, initial_state=hf)
    return pass_mgr.run(raw)


def run_vqe(
    qubit_op,
    ansatz,
    estimator,
    optimizer,
    initial_point: np.ndarray,
) -> tuple[float, np.ndarray]:
    vqe = VQE(estimator, ansatz, optimizer, initial_point=initial_point)
    result = vqe.compute_minimum_eigenvalue(qubit_op)
    return float(result.eigenvalue.real), np.asarray(result.optimal_point, dtype=float)


# ---------------------------------------------------------------------------
# Sanity check: exact diag of the active-space qubit Hamiltonian
# ---------------------------------------------------------------------------
def exact_ground_state(qubit_op) -> float:
    H = qubit_op.to_matrix()
    return float(np.linalg.eigvalsh(H)[0])


# ---------------------------------------------------------------------------
# Main sweep + plots
# ---------------------------------------------------------------------------
def main() -> None:
    distances = np.array([0.7, 1.0, 1.3, 1.6, 2.0, 2.5, 3.0, 3.5, 4.0])

    mapper = JordanWignerMapper()
    estimator = AerEstimator()
    optimizer = COBYLA(maxiter=400, rhobeg=0.5)
    pass_mgr = generate_preset_pass_manager(
        optimization_level=0, backend=AerSimulator()
    )

    e_hf, e_fci, e_vqe, e_diag = [], [], [], []
    initial_pt: np.ndarray | None = None
    ansatz = None

    for d in distances:
        t0 = time.time()
        data = run_pyscf(float(d))

        h1_eff, eri_act, e_inactive, n_active = freeze_core(
            data["h1_mo"], data["eri_mo"], data["e_nuc"],
            data["n_orb"], N_FROZEN_CORE,
        )
        n_active_elec = data["n_elec"] - 2 * N_FROZEN_CORE
        n_alpha = n_active_elec // 2
        n_particles = (n_alpha, n_active_elec - n_alpha)

        ferm_op = build_fermionic_hamiltonian(h1_eff, eri_act, n_active)
        qubit_op = mapper.map(ferm_op)

        # Active-space FCI (exact diag of the qubit Hamiltonian); the gap to
        # the full FCI is the small frozen-core error.
        e_active_diag = exact_ground_state(qubit_op) + e_inactive

        # Active space and ansatz topology don't change across distances --
        # build the ansatz once and reuse it.
        if ansatz is None:
            ansatz = make_ansatz(n_active, n_particles, mapper, pass_mgr)
            initial_pt = np.zeros(ansatz.num_parameters)

        e_active_vqe, initial_pt = run_vqe(
            qubit_op, ansatz, estimator, optimizer, initial_pt
        )
        e_total_vqe = e_active_vqe + e_inactive

        e_hf.append(data["e_hf"])
        e_fci.append(data["e_fci"])
        e_vqe.append(e_total_vqe)
        e_diag.append(e_active_diag)

        dt = time.time() - t0
        print(
            f"d = {d:4.2f} A  |  HF = {data['e_hf']:.6f}  "
            f"FCI = {data['e_fci']:.6f}  diag_act = {e_active_diag:.6f}  "
            f"VQE = {e_total_vqe:.6f}  "
            f"(VQE-FCI = {(e_total_vqe - data['e_fci']) * 1e3:+.3f} mEh, "
            f"{dt:.1f}s)"
        )

    e_hf = np.asarray(e_hf)
    e_fci = np.asarray(e_fci)
    e_vqe = np.asarray(e_vqe)

    # --- Dissociation curve ------------------------------------------------
    plt.figure(figsize=(8, 5))
    plt.plot(distances, e_hf, "o-", label="Hartree-Fock")
    plt.plot(distances, e_fci, "s-", label="FCI (PySCF, full space)")
    plt.plot(distances, e_vqe, "^--", label="VQE (UCCSD / Jordan-Wigner)")
    plt.xlabel("Li-H distance (Angstrom)")
    plt.ylabel("Total energy (Hartree)")
    plt.title("LiH dissociation curve, STO-3G")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("dissociation_curve.png", dpi=150)
    plt.close()

    # --- Correlation energy ------------------------------------------------
    e_corr_fci = (e_fci - e_hf) * 1000.0
    e_corr_vqe = (e_vqe - e_hf) * 1000.0

    plt.figure(figsize=(8, 5))
    plt.plot(distances, e_corr_fci, "s-", label=r"$E_{\rm FCI} - E_{\rm HF}$")
    plt.plot(distances, e_corr_vqe, "^--", label=r"$E_{\rm VQE} - E_{\rm HF}$")
    plt.axhline(0.0, color="black", linewidth=0.5)
    plt.xlabel("Li-H distance (Angstrom)")
    plt.ylabel("Correlation energy (mHartree)")
    plt.title("LiH correlation energy across dissociation, STO-3G")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("correlation_energy.png", dpi=150)
    plt.close()

    print("\nSaved dissociation_curve.png and correlation_energy.png")


if __name__ == "__main__":
    main()
