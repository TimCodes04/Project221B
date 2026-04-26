# Code walk-through: `lih_vqe.py`

This document walks through every section of [lih_vqe.py](lih_vqe.py) — what
it computes, the math behind it, and why each piece is there. It is meant to
be read alongside the source.

The file is structured top-to-bottom in the order the pipeline runs:

1. PySCF classical pre-processing → integrals + reference energies.
2. Frozen-core projection → effective integrals on the active space.
3. Slater–Condon construction → second-quantized `FermionicOp`.
4. Jordan–Wigner mapping → qubit `SparsePauliOp`.
5. UCCSD ansatz built on top of Hartree–Fock, transpiled for Aer.
6. VQE with COBYLA, with bootstrapped initial points.
7. Plots.

---

## 0. Imports and constants ([lih_vqe.py:21-40](lih_vqe.py:21))

```python
from pyscf import ao2mo, fci, gto, scf
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import EstimatorV2 as AerEstimator
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import COBYLA
from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.operators import FermionicOp

N_FROZEN_CORE = 1   # number of doubly occupied spatial orbitals to freeze
INTEGRAL_TOL  = 1e-12
```

The PySCF imports do classical electronic structure (build the molecule, run
SCF, run FCI, get integrals). The Qiskit imports do quantum simulation:
`FermionicOp` is the second-quantized operator class, `JordanWignerMapper`
turns it into a qubit operator, `HartreeFock` and `UCCSD` provide circuit
templates, and `VQE` + `COBYLA` form the hybrid classical–quantum loop.
`AerEstimator` evaluates $\langle \Psi(\theta)|\hat H|\Psi(\theta)\rangle$
on a noiseless statevector simulator.

`N_FROZEN_CORE = 1` means we freeze the lowest spatial orbital (the Li 1s
core) before doing anything quantum. `INTEGRAL_TOL` is the threshold below
which a one- or two-electron integral is treated as zero — purely a sparsity
optimization for the operator dictionary.

---

## 1. Classical pre-processing — `run_pyscf` ([lih_vqe.py:46](lih_vqe.py:46))

```python
def run_pyscf(distance: float) -> dict:
    mol = gto.M(atom=f"Li 0 0 0; H 0 0 {distance}", basis="sto-3g", ...)
    mf  = scf.RHF(mol); mf.kernel()
    h1_ao  = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
    h1_mo  = mf.mo_coeff.T @ h1_ao @ mf.mo_coeff
    eri_mo = ao2mo.kernel(mol, mf.mo_coeff, compact=False).reshape(n_orb, n_orb, n_orb, n_orb)
    e_fci, _ = fci.FCI(mf).kernel()
```

We build the molecule in the **STO-3G** minimal basis: 5 spatial orbitals
on Li ($1s, 2s, 2p_x, 2p_y, 2p_z$) and 1 on H ($1s$), so $N_\text{orb} = 6$
and $2 N_\text{orb} = 12$ spin orbitals. LiH has 4 electrons (Li: 3, H: 1),
giving a closed-shell singlet, so RHF applies.

After SCF converges, `mf.mo_coeff` is the matrix $C$ whose columns are the
molecular-orbital (MO) coefficients in the AO basis. We need three things in
the MO basis:

- **One-electron MO integrals** $h_{pq} = \langle \phi_p | \hat T + \hat
  V_\text{nuc}|\phi_q\rangle$. Computed in the AO basis via PySCF's
  `int1e_kin` + `int1e_nuc` and rotated to the MO basis as $C^\top h^\text{AO}
  C$.
- **Two-electron MO integrals** $(pq|rs)$ in **chemist's notation**:
  $$ (pq|rs) \;=\; \int\!\!\int \phi_p^*(1)\phi_q(1)\,\frac{1}{r_{12}}\,\phi_r^*(2)\phi_s(2)\,d\mathbf r_1\,d\mathbf r_2. $$
  PySCF's `ao2mo.kernel(..., compact=False)` returns these as a
  $N_\text{orb}^4$ tensor.
- **Full-space FCI energy** as the reference "exact" answer for comparison
  later.

Everything is bundled into a dict and returned. This function is called
once per Li–H distance.

> **Sign / unit conventions.** Hartree atomic units throughout. The 4-index
> tensor in chemist's notation has the symmetries $(pq|rs) = (qp|rs) =
> (pq|sr) = (rs|pq)$ (real orbitals), which is why we don't have to worry
> about complex conjugation.

---

## 2. Frozen-core projection — `freeze_core` ([lih_vqe.py:80](lih_vqe.py:80))

The Li 1s orbital is so tightly bound and energetically separated that
correlating its electrons with the valence ones contributes only ~µHartree
to the total. Freezing it analytically cuts the spin-orbital count from 12
to 10 and the active-electron count from 4 to 2.

The standard frozen-core algebra works as follows. Partition orbital indices
into **inactive** $\{i, j\}$ (frozen, doubly occupied) and **active** $\{p,
q, r, s\}$. The full Hamiltonian acting on the active space (with the
inactive electrons traced out) becomes

$$
\hat H \;=\; E_\text{inactive}
\;+\; \sum_{pq} \tilde h_{pq} \sum_\sigma a^\dagger_{p\sigma} a_{q\sigma}
\;+\; \tfrac{1}{2}\sum_{pqrs} (pq|rs) \sum_{\sigma\tau}
       a^\dagger_{p\sigma} a^\dagger_{r\tau} a_{s\tau} a_{q\sigma},
$$

where the inactive constant and the effective one-body operator are

$$
E_\text{inactive} \;=\; E_\text{nuc}
   + 2\sum_i h_{ii}
   + \sum_{ij}\bigl[\,2(ii|jj) - (ij|ji)\,\bigr],
$$

$$
\tilde h_{pq} \;=\; h_{pq}
   + \sum_i \bigl[\,2(pq|ii) - (pi|iq)\,\bigr].
$$

Both are direct consequences of evaluating closed-shell expectation values
of the original Hamiltonian. The factor of 2 in front of $h_{ii}$ counts
both spins of the doubly-occupied core; the $2J - K$ structure is the
familiar Coulomb minus exchange contribution from the core to either an
active electron ($\tilde h_{pq}$) or another core electron
($E_\text{inactive}$).

The function ([lih_vqe.py:80](lih_vqe.py:80)) implements those two formulas
literally:

```python
e_inactive = e_nuc + 2 * sum(h1[i, i] for i in inactive)
e_inactive += sum(2 * eri[i,i,j,j] - eri[i,j,j,i] for i,j in inactive×inactive)

h1_eff = h1[active, active].copy()
h1_eff[p, q] += sum(2 * eri[P,Q,i,i] - eri[P,i,i,Q] for i in inactive)

eri_act = eri[active, active, active, active]
```

Returned: `(h1_eff, eri_act, e_inactive, n_active)`. `e_inactive` is a
*scalar* that we'll add back to the VQE result at the end so totals are
reported on the same footing as PySCF's full-space FCI.

---

## 3. Slater–Condon construction — `build_fermionic_hamiltonian` ([lih_vqe.py:122](lih_vqe.py:122))

This is the heart of the "constructed via the Slater–Condon rules" line in
the proposal. We take the active-space integrals and emit a `FermionicOp`
in spin-orbital second quantization.

### 3.1 Spin-orbital ordering

Qiskit Nature's `FermionicOp` indexes spin orbitals with a single integer.
The convention used here is the standard "block-spin" ordering:

```
spin orbital index i = p             for alpha (spin-up)   spatial orbital p
                       p + n_orb     for beta  (spin-down) spatial orbital p
```

So with `n_orb = 5` active spatial orbitals, indices 0–4 are alpha and 5–9
are beta.

### 3.2 The one-electron block

```python
for p, q in itertools.product(range(n_orb), repeat=2):
    v = h1[p, q]
    for off in (0, n_orb):
        op[f"+_{p+off} -_{q+off}"] += v
```

This builds $\sum_{pq\sigma} h_{pq}\, a^\dagger_{p\sigma} a_{q\sigma}$.
The `off` loop sums over $\sigma \in \{\alpha, \beta\}$ — the spatial
integrals are spin-independent, so each $h_{pq}$ contributes identically
to both spin sectors.

### 3.3 The two-electron block

In second quantization the two-body part of the Hamiltonian, written in
**physicist's notation** $\langle pq | rs\rangle$, is

$$
\hat V_2 = \tfrac{1}{2}\sum_{pqrs}\langle pq|rs\rangle\,
  a^\dagger_p a^\dagger_q a_s a_r.
$$

The relation between physicist's and chemist's notation is
$\langle pq | rs\rangle = (pr | qs)$, which after relabelling
($q \leftrightarrow r$ in the dummy sum) yields the form we actually use:

$$
\hat V_2 \;=\; \tfrac{1}{2}\sum_{pqrs}(pq|rs)\,
   a^\dagger_p\,a^\dagger_r\,a_s\,a_q.
$$

Adding the spin sums (spatial integrals are diagonal in spin) gives

$$
\hat V_2 = \tfrac{1}{2}\sum_{pqrs}(pq|rs)\sum_{\sigma\tau}
   a^\dagger_{p\sigma}\,a^\dagger_{r\tau}\,a_{s\tau}\,a_{q\sigma}.
$$

The code is a direct transcription:

```python
for p, q, r, s in itertools.product(range(n_orb), repeat=4):
    v = eri[p, q, r, s]
    half = 0.5 * v
    for s_off, t_off in itertools.product((0, n_orb), repeat=2):  # σ, τ
        ip, iq = p + s_off, q + s_off          # creators / annihilators on σ
        ir, is_ = r + t_off, s + t_off          # creators / annihilators on τ
        if ip == ir or iq == is_:
            continue                            # vanishes by Pauli exclusion
        op[f"+_{ip} +_{ir} -_{is_} -_{iq}"] += half
```

The Pauli-exclusion skip handles cases where the operator literally
evaluates to zero. With $\sigma = \tau$ and $p = r$, we'd have $a^\dagger_{p\sigma}
a^\dagger_{p\sigma} = 0$; same logic for $a_{s\tau}a_{q\sigma}$ when
$\sigma = \tau$ and $s = q$. (When $\sigma \ne \tau$, even if $p = r$ in
spatial-orbital index, the spin orbitals are distinct, so the operator
doesn't vanish.) Skipping these terms isn't strictly necessary — they would
contribute zero — but it keeps the operator dictionary clean.

### 3.4 String-format for FermionicOp

Qiskit Nature's `FermionicOp` accepts a `dict` of label → coefficient. The
label `"+_3 -_5"` means $a^\dagger_3 a_5$, with operators ordered
left-to-right *as written in the string*. Acting on a state, the rightmost
operator applies first (standard math convention for operator products).

The constant $E_\text{inactive}$ is **not** included in this `FermionicOp`.
We add it back in `main` after VQE returns. Either approach is valid — keeping
the constant out of the operator just makes the qubit-Hamiltonian matrix a
bit cleaner.

---

## 4. Jordan–Wigner mapping ([lih_vqe.py:241](lih_vqe.py:241))

```python
mapper   = JordanWignerMapper()
qubit_op = mapper.map(ferm_op)
```

The Jordan–Wigner transform sends each spin orbital to a qubit:

$$
a_j \;\to\; \tfrac{1}{2}(X_j + iY_j)\,\prod_{k<j}Z_k,
\qquad
a^\dagger_j \;\to\; \tfrac{1}{2}(X_j - iY_j)\,\prod_{k<j}Z_k.
$$

The Z-string on $k < j$ enforces the fermionic anticommutation relations
on the qubit register. After mapping, the active-space Hamiltonian is a
`SparsePauliOp` on 10 qubits. For LiH STO-3G frozen-core, that's a few
hundred Pauli strings, each evaluable on the statevector estimator in O(2^10)
time.

> JW is the simplest mapping; it has linear-weight one-body terms but
> O($n$)-weight strings on the Z-tail. Alternatives like Bravyi–Kitaev or
> parity (with tapering) reduce the Pauli weight or qubit count, but the
> proposal specifies JW so we stick with it.

---

## 5. The ansatz: HF + UCCSD, transpiled for Aer — `make_ansatz` ([lih_vqe.py:168](lih_vqe.py:168))

```python
def make_ansatz(n_active_orb, n_particles, mapper, pass_mgr):
    hf  = HartreeFock(n_active_orb, n_particles, mapper)
    raw = UCCSD(n_active_orb, n_particles, mapper, initial_state=hf)
    return pass_mgr.run(raw)
```

### 5.1 Hartree–Fock initial state

`HartreeFock(n_active_orb, n_particles, mapper)` produces a Pauli-X-only
circuit that flips qubits corresponding to occupied spin orbitals in the HF
reference. With 1 alpha + 1 beta active electron and JW ordering, the HF
state is $|0000010001\rangle$ (alpha occ on qubit 0, beta occ on qubit 5;
read with qubit 0 as the rightmost bit).

### 5.2 UCCSD ansatz

UCCSD parameterises the wavefunction as

$$
|\Psi(\theta)\rangle \;=\; e^{\hat T(\theta) - \hat T^\dagger(\theta)}\,|\Phi_\text{HF}\rangle,
\qquad
\hat T = \hat T_1 + \hat T_2,
$$

with single and double excitations

$$
\hat T_1 = \sum_{ia} t_i^a\, a^\dagger_a a_i,
\qquad
\hat T_2 = \tfrac{1}{4}\sum_{ijab} t_{ij}^{ab}\, a^\dagger_a a^\dagger_b a_j a_i,
$$

where $i, j$ run over occupied and $a, b$ over virtual orbitals. The
unitary $U(\theta) = \exp(\hat T - \hat T^\dagger)$ preserves particle
number and (with appropriate parameter tying) total spin.

Qiskit Nature's `UCCSD` builds this by:

1. Enumerating all symmetry-allowed singles and doubles.
2. Mapping each $\hat T_k - \hat T_k^\dagger$ to a Pauli operator via the
   chosen mapper (JW here).
3. Trotterising — by default with 1 step — into a product of
   `PauliEvolutionGate`s, one per excitation.

Each `PauliEvolutionGate` carries one variational parameter $\theta_k$.

### 5.3 Why we transpile (the `EvolvedOps` story)

`UCCSD` represents itself with a custom Qiskit instruction called
`EvolvedOps`. Aer's circuit assembler doesn't recognise that opcode —
running VQE on the raw ansatz crashes with
`AerError: 'unknown instruction: EvolvedOps'`. The fix is to transpile the
ansatz once into Aer's native gate set:

```python
pass_mgr = generate_preset_pass_manager(optimization_level=0,
                                        backend=AerSimulator())
ansatz   = pass_mgr.run(raw)
```

This expands every `EvolvedOps` into the underlying `PauliEvolutionGate`s
and then into elementary single- and two-qubit gates that Aer understands.
We do this **once**, not per VQE iteration, because the active space (and
hence the ansatz topology) doesn't change between distances — only the
parameter values do.

> Aside: `qiskit.primitives.StatevectorEstimator` (the reference V2
> estimator outside Aer) accepts the raw `EvolvedOps` directly and would
> avoid this transpile entirely. But the proposal asks for the Aer
> simulator, so we go through `AerEstimator` and transpile.

---

## 6. The VQE call — `run_vqe` ([lih_vqe.py:180](lih_vqe.py:180))

```python
def run_vqe(qubit_op, ansatz, estimator, optimizer, initial_point):
    vqe    = VQE(estimator, ansatz, optimizer, initial_point=initial_point)
    result = vqe.compute_minimum_eigenvalue(qubit_op)
    return float(result.eigenvalue.real), np.asarray(result.optimal_point)
```

`qiskit_algorithms.VQE` expects a V2 estimator (it builds PUBs of the form
`(circuit, observable, parameter_values)` and calls `estimator.run([pub])`).
On each COBYLA step it:

1. Asks COBYLA for the next $\theta$.
2. Builds the PUB `(ansatz, qubit_op, θ)` and submits to the estimator.
3. Returns the expectation value $\langle \Psi(\theta)|\hat H|\Psi(\theta)\rangle$.
4. COBYLA decides whether to accept and proposes the next $\theta$.

We use `COBYLA(maxiter=400, rhobeg=0.5)`. `rhobeg` is the initial trust-
region radius — 0.5 radians is a reasonable per-parameter step for UCCSD
parameters that typically end up small (often $|\theta| \lesssim 0.1$). The
`maxiter` cap is generous; convergence is usually well under that.

The function returns the optimised energy *of the active-space qubit
Hamiltonian*. The total molecular energy is recovered later as
$E_\text{total} = E_\text{active} + E_\text{inactive}$.

---

## 7. Sanity check — `exact_ground_state` ([lih_vqe.py:195](lih_vqe.py:195))

```python
def exact_ground_state(qubit_op):
    H = qubit_op.to_matrix()
    return float(np.linalg.eigvalsh(H)[0])
```

For 10 qubits the dense Hamiltonian matrix is $1024 \times 1024$ — trivial
to diagonalize with NumPy. This gives the **exact ground-state energy of the
active-space Hamiltonian** and tells us, for each distance, two things:

- $E_\text{diag} + E_\text{inactive}$ vs. PySCF's full-space FCI: the
  **frozen-core error**.
- $E_\text{VQE}$ vs. $E_\text{diag} + E_\text{inactive}$: the **VQE
  convergence error**.

Both are reported in the per-distance log line and they're a clean way to
attribute error: in our run the convergence error is a few µHartree at
every point, so essentially all of the small VQE-vs-FCI gap is the frozen
core, not VQE.

---

## 8. Main loop — `main` ([lih_vqe.py:203](lih_vqe.py:203))

### 8.1 Setup

```python
distances = np.array([0.7, 1.0, 1.3, 1.6, 2.0, 2.5, 3.0, 3.5, 4.0])
mapper    = JordanWignerMapper()
estimator = AerEstimator()
optimizer = COBYLA(maxiter=400, rhobeg=0.5)
pass_mgr  = generate_preset_pass_manager(optimization_level=0,
                                         backend=AerSimulator())
```

Distance grid concentrates points near equilibrium ($\sim$1.6 Å) and spans
out to 4.0 Å where dissociation is essentially complete. The estimator,
optimizer, mapper, and pass manager are *stateless across distances* — we
build them once and reuse.

### 8.2 Per-distance work

Inside the loop:

1. `run_pyscf(d)` — RHF, FCI, integrals.
2. `freeze_core(...)` — project to the 5-orbital, 2-electron active space.
3. `build_fermionic_hamiltonian(...)` — Slater–Condon →
   `FermionicOp`.
4. `mapper.map(...)` — Jordan–Wigner → `SparsePauliOp`.
5. `exact_ground_state(...)` — NumPy diag for the sanity baseline.
6. `make_ansatz(...)` — built **once** on the first iteration and reused
   thereafter (the active space and `n_particles` are distance-independent).
7. `run_vqe(...)` — COBYLA-driven optimization.
8. Add `e_inactive` back to recover the total energy and append to the
   plotting arrays.

### 8.3 Bootstrapping the initial point

```python
e_active_vqe, initial_pt = run_vqe(qubit_op, ansatz, estimator,
                                   optimizer, initial_pt)
```

The VQE return includes the optimised parameter vector. We feed it directly
into the next distance as `initial_point`, so each subsequent COBYLA run
starts from a known good $\theta^\star$ instead of zeros. This typically
cuts the number of energy evaluations needed for convergence by 2–5×, and
it works here because the *topology* of $\theta$-space is the same for
every distance (same ansatz, same parameter count) — only the optimum
shifts smoothly with $r$.

The first call seeds with `np.zeros(...)`, which corresponds to the
$\hat T = 0$ → pure HF state.

---

## 9. Plots ([lih_vqe.py:254](lih_vqe.py:254))

Two figures are saved to disk:

**`dissociation_curve.png`** — $E_\text{HF}$, $E_\text{FCI}$, $E_\text{VQE}$
vs. $r_\text{Li-H}$. HF tracks FCI well near equilibrium and diverges
upward as the bond stretches. FCI and VQE overlap over the full range.

**`correlation_energy.png`** — $E - E_\text{HF}$ in mHartree for both FCI
and VQE. Both curves overlap; the depth at $r = 4$ Å (~$-160$ mHartree)
is the magnitude of static correlation that HF is missing in the
dissociated regime, and the central qualitative result of the project is
that VQE captures it.

---

## 10. Wall-clock summary

End-to-end runtime on a laptop is ~2:40. Per-distance breakdown
(approximate):

| Step | Time |
| --- | --- |
| PySCF (RHF + FCI + integrals) | ~50 ms |
| Frozen-core projection | <1 ms |
| Slater–Condon → `FermionicOp` | ~10 ms |
| Jordan–Wigner mapping | ~30 ms |
| Exact diag (sanity) | ~50 ms |
| **VQE (COBYLA + AerEstimator, ~150 iters)** | **~16 s** |
| Total per distance | ~17 s |

VQE dominates — every other step is essentially free at this system size.
Each VQE iteration costs one transpiled-ansatz statevector simulation plus
one expectation-value evaluation against ~few-hundred Pauli terms.
