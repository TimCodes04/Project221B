# LiH ground-state correlation energy via VQE

CHEM 221B project. Code computes the ground-state energy of LiH
along its bond-dissociation coordinate using a Variational Quantum Eigensolver
(VQE) with a Unitary Coupled-Cluster Singles–Doubles (UCCSD) ansatz, and
compare against Hartree–Fock and Full Configuration Interaction (FCI)
references obtained from PySCF. All circuits run on Qiskit Aer's statevector
simulator (no shot noise).

The interesting physics: near equilibrium ($r_\text{Li-H} \approx 1.6$ Å) a
single Slater determinant captures most of the energy, so HF is already very
close to FCI and the correlation energy
$E_\text{corr} = E_\text{FCI} - E_\text{HF}$ is small (~17 mHartree). As the
bond stretches, HF degrades quickly because it cannot describe the multi-
configurational character of the dissociated system, and $E_\text{corr}$ grows
to ~160 mHartree at $r = 4.0$ Å. The central result of this project is that
**VQE with UCCSD tracks FCI all the way through dissociation**, while HF
diverges from it.

## Pipeline

1. **PySCF** — RHF + FCI + one- and two-electron MO integrals at each Li–H
   distance.
2. **Frozen-core projection** — analytically integrate out the Li 1s core,
   producing an inactive energy and an effective $\tilde h_{pq}$ on the
   remaining 5 spatial orbitals.
3. **Slater–Condon → second quantization** — build the active-space
   Hamiltonian
   $\hat H = \sum_{pq\sigma} \tilde h_{pq}\, a^\dagger_{p\sigma} a_{q\sigma}
       + \tfrac{1}{2}\sum_{pqrs\sigma\tau} (pq|rs)\, a^\dagger_{p\sigma}
         a^\dagger_{r\tau} a_{s\tau} a_{q\sigma}$
   directly from the MO integrals.
4. **Jordan–Wigner mapping** — the active-space `FermionicOp` becomes a
   10-qubit `SparsePauliOp`.
5. **UCCSD ansatz on top of Hartree–Fock** — built from `qiskit_nature`'s
   library and transpiled to Aer's native gate set so that the
   `EvolvedOps` instruction is decomposed into elementary rotations.
6. **VQE with COBYLA** — minimise $\langle \Psi(\theta)|\hat H|\Psi(\theta)\rangle$,
   bootstrapping the previous distance's optimum as the initial point for the
   next.
7. **Plot** dissociation curve and correlation-energy curve.


>>>>>>> b1d4d6e921fed2116f911dfefb6e3b1e99f9afa5
## How to run

The included `venv/` already has every dependency pinned. From the project
root:

```bash
./venv/bin/python lih_vqe.py
```

End-to-end runtime is ~2–3 minutes on a laptop. The script writes the two
PNGs and prints a per-distance summary line, e.g.

```
d = 1.60 A  |  HF = -7.861865  FCI = -7.882324  diag_act = -7.882097  VQE = -7.882097  (VQE-FCI = +0.228 mEh, 16.8s)
```

The `diag_act` column is the exact diagonalization of the active-space qubit
Hamiltonian — it acts as a sanity check that `VQE ≈ diag_act` at every point.

## Dependencies

Pinned versions (Python 3.13, all installed in `venv/`):

| Package | Version |
| --- | --- |
| qiskit | 1.4.5 |
| qiskit-aer | 0.17.2 |
| qiskit-algorithms | 0.4.0 |
| qiskit-nature | 0.7.2 |
| pyscf | 2.12.1 |
| numpy | 2.4.4 |
| matplotlib | 3.10.8 |

## Sample numbers (STO-3G)

| $r_\text{Li-H}$ (Å) | $E_\text{HF}$ | $E_\text{FCI}$ | $E_\text{VQE}$ | $E_\text{VQE} - E_\text{FCI}$ |
| ---: | ---: | ---: | ---: | ---: |
| 0.7 | -7.485945 | -7.505052 | -7.503999 | +1.05 mEh |
| 1.0 | -7.767362 | -7.784460 | -7.784021 | +0.44 mEh |
| 1.3 | -7.851954 | -7.869140 | -7.868904 | +0.24 mEh |
| 1.6 | -7.861865 | -7.882324 | -7.882097 | +0.23 mEh |
| 2.0 | -7.830906 | -7.861088 | -7.860828 | +0.26 mEh |
| 2.5 | -7.770874 | -7.823724 | -7.823413 | +0.31 mEh |
| 3.0 | -7.710830 | -7.798843 | -7.798500 | +0.34 mEh |
| 3.5 | -7.661202 | -7.788115 | -7.787753 | +0.36 mEh |
| 4.0 | -7.624976 | -7.784278 | -7.783927 | +0.35 mEh |

The residual $E_\text{VQE} - E_\text{FCI}$ is dominated by the
**frozen-core approximation** (Li 1s removed from the correlated space), not
by VQE convergence — VQE matches the exact diagonalization of the
active-space Hamiltonian to within ~10 µHartree at every point.

## Notes / caveats

- **Why `qiskit.primitives.StatevectorEstimator` is *not* used.**  Aer's
  `EstimatorV2` does not understand the `EvolvedOps` instruction emitted by
  `qiskit_nature`'s UCCSD; the script works around this by transpiling the
  ansatz once into Aer's native gate set before VQE starts (see
  [`make_ansatz`](lih_vqe.py:168)).
- **Frozen core.** Freezing the Li 1s drops the qubit count from 12 to 10 and
  gives a ~10× speedup with sub-mHartree error. The frozen-core energy is
  computed analytically and added back after VQE so totals are reported on
  the same footing as PySCF's full-space FCI.
- **Bootstrapping.** The optimal $\theta^\star$ from one distance seeds the
  next, which keeps COBYLA from re-exploring the parameter landscape from
  scratch. The active space is fixed across distances, so `num_parameters`
  doesn't change.
