"""
Coarse-Grained Molecular Dynamics Engine
==========================================

Builds and runs coarse-grained (CG) MD simulations for a described material,
for use by `md_agent.py`. Two model families, one shared potential form:

  * "protein" — HPS model (Dignon et al., PLoS Comput Biol 2018): one bead
    per residue, harmonic bonds, Ashbaugh-Hatch hydropathy-scaled
    short-range pairs + Debye-Huckel screened electrostatics, implicit
    solvent. Real units: nm, ps, amu, kJ/mol, K.
  * "polymer" — generic bead-spring homopolymer (Kremer-Grest-like, with
    harmonic instead of FENE bonds) in reduced LJ units (sigma = 1, epsilon
    = 1, m = 1, kT = 1). Solvent quality is set by the Ashbaugh-Hatch
    attraction lambda: 0 = purely repulsive WCA (good solvent), 1 = full LJ
    (poor solvent at kT = 1).

Two backends implement the same Hamiltonian:

  * "numpy"  — dependency-free (numpy only) reference implementation:
    Verlet neighbor list + BAOAB Langevin integrator. Fine for ~1000 beads
    and ~1e5 steps; slow beyond that.
  * "openmm" — the same potential via CustomNonbondedForce, with OpenMM's
    LangevinMiddleIntegrator (also BAOAB). Much faster; used automatically
    when `openmm` is importable.

The validation harness (`python3 md_engine.py`) checks finite-difference
forces, numpy-vs-OpenMM energy/force agreement, equipartition, ideal-chain
statistics, and good-vs-poor solvent collapse — same pattern as
`bio_tools.py`: real algorithms checked against known physics, not stubs.

What this model can't tell you (be explicit about this downstream): one
bead per residue means no secondary structure, no beta-sheet nanocrystals,
no hydrogen bonds; implicit solvent + Langevin friction means dynamical
quantities (diffusion, relaxation times) are model-time, not real time.
"""

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

KB_KJ_PER_MOL_K = 0.0083144626      # Boltzmann constant, kJ/(mol K)
COULOMB_KJ_NM = 138.935458          # e^2 / (4 pi eps0), kJ nm / mol
AVOGADRO = 6.02214076e23

# ---------------------------------------------------------------------------
# HPS residue parameters (Dignon et al. 2018, Kapcha-Rossky hydropathy scale
# normalized to [0, 1]). sigma in Angstrom (converted to nm at build time),
# standard residue masses in amu. These values are transcribed from the
# published HPS parameter table — spot-check them against the paper's SI
# before relying on absolute numbers.
# ---------------------------------------------------------------------------

HPS_PARAMS: Dict[str, Tuple[float, float, float]] = {
    # residue: (mass_amu, sigma_A, lambda)
    "A": (71.08, 5.04, 0.730),
    "R": (156.20, 6.56, 0.000),
    "N": (114.10, 5.68, 0.432),
    "D": (115.10, 5.58, 0.378),
    "C": (103.10, 5.48, 0.595),
    "Q": (128.10, 6.02, 0.514),
    "E": (129.10, 5.92, 0.459),
    "G": (57.05, 4.50, 0.649),
    "H": (137.10, 6.08, 0.514),
    "I": (113.20, 6.18, 0.973),
    "L": (113.20, 6.18, 0.973),
    "K": (128.20, 6.36, 0.514),
    "M": (131.20, 6.18, 0.838),
    "F": (147.20, 6.36, 1.000),
    "P": (97.12, 5.56, 1.000),
    "S": (87.08, 5.18, 0.595),
    "T": (101.10, 5.62, 0.676),
    "W": (186.20, 6.78, 0.946),
    "Y": (163.20, 6.46, 0.865),
    "V": (99.07, 5.86, 0.892),
}
HPS_EPSILON = 0.8368          # kJ/mol (0.2 kcal/mol)
HPS_BOND_LENGTH = 0.38        # nm
HPS_BOND_K = 8033.0           # kJ/(mol nm^2), U = k/2 (r - r0)^2
HPS_LJ_CUTOFF = 2.0           # nm
HPS_DH_CUTOFF = 3.5           # nm
HPS_DIELECTRIC = 80.0

POLYMER_BOND_LENGTH = 1.0     # sigma
POLYMER_BOND_K = 500.0        # epsilon / sigma^2
POLYMER_LJ_CUTOFF = 2.5       # sigma
SOLVENT_QUALITY_LAMBDA = {
    "good": 0.0,
    # Approximate: full-LJ Kremer-Grest chains have a theta temperature of
    # roughly kT ~ 3 epsilon, so scaling attraction by ~1/3 at kT = 1 lands
    # near theta. Not separately calibrated here — treat as approximate.
    "theta": 0.35,
    "poor": 1.0,
}

MAX_BEADS = 5000              # hard safety limit for agent-built systems


class SpecError(ValueError):
    """Raised when a MaterialSpec is invalid or outside what the model supports.
    The agent surfaces this message back to the LLM so it can correct the spec."""


class BackendNotAvailableError(RuntimeError):
    """Raised when a requested backend (e.g. openmm) isn't installed — distinct
    from a simulation failure so callers can fall back instead of retrying."""


# ---------------------------------------------------------------------------
# Spec and system
# ---------------------------------------------------------------------------

@dataclass
class MaterialSpec:
    """Structured, validated description of what to simulate."""
    model: str                                  # "protein" | "polymer"
    name: str = "material"
    sequence: Optional[str] = None              # protein only
    chain_length: Optional[int] = None          # polymer only
    solvent_quality: str = "good"               # polymer only: good|theta|poor
    attraction_lambda: Optional[float] = None   # polymer only: overrides solvent_quality
    n_chains: int = 1
    temperature_K: float = 300.0                # protein only (polymer is reduced kT = 1)
    ionic_strength_M: float = 0.15              # protein only
    histidine_charge: float = 0.0               # protein only (0 ~ pH 7; 0.5 ~ pH 6)
    concentration_mM: Optional[float] = None    # multi-chain protein: sets box size
    box_length: Optional[float] = None          # explicit cubic box edge (nm or sigma)
    seed: int = 0

    def validate(self) -> None:
        if self.model not in ("protein", "polymer"):
            raise SpecError(f"model must be 'protein' or 'polymer', got {self.model!r}")
        if self.n_chains < 1:
            raise SpecError("n_chains must be >= 1")
        if self.model == "protein":
            if not self.sequence:
                raise SpecError("protein model requires a sequence")
            seq = "".join(self.sequence.split()).upper()
            bad = sorted(set(seq) - set(HPS_PARAMS))
            if bad:
                raise SpecError(f"sequence contains non-standard residue codes {bad}; "
                                "the HPS model only parameterizes the 20 standard amino acids")
            if len(seq) < 5:
                raise SpecError("sequence too short (< 5 residues) to define chain properties")
            self.sequence = seq
            if not (250.0 <= self.temperature_K <= 450.0):
                raise SpecError("temperature_K outside 250-450 K, where HPS is not meaningful")
            if not (0.0 <= self.ionic_strength_M <= 2.0):
                raise SpecError("ionic_strength_M must be in [0, 2]")
        else:
            if not self.chain_length or self.chain_length < 5:
                raise SpecError("polymer model requires chain_length >= 5")
            if self.attraction_lambda is None:
                if self.solvent_quality not in SOLVENT_QUALITY_LAMBDA:
                    raise SpecError(f"solvent_quality must be one of {list(SOLVENT_QUALITY_LAMBDA)}")
            elif not (0.0 <= self.attraction_lambda <= 1.0):
                raise SpecError("attraction_lambda must be in [0, 1]")
        n = self.n_beads()
        if n > MAX_BEADS:
            raise SpecError(f"system would have {n} beads (> {MAX_BEADS} limit); "
                            "reduce n_chains or chain length")

    def n_beads(self) -> int:
        per_chain = len(self.sequence) if self.model == "protein" else int(self.chain_length)
        return per_chain * self.n_chains


@dataclass
class ParticleSystem:
    """Everything the backends need. Unit system is consistent per `units`."""
    units: str                       # "real (nm, ps, amu, kJ/mol)" or "reduced (LJ)"
    length_unit: str                 # "nm" or "sigma"
    kT: float                        # energy units
    masses: np.ndarray               # (N,)
    sigma: np.ndarray                # (N,)
    lam: np.ndarray                  # (N,)
    charge: np.ndarray               # (N,)
    epsilon: float
    bonds: np.ndarray                # (B, 2) int
    bond_r0: float
    bond_k: float
    lj_cutoff: float
    dh_cutoff: float
    debye_length: Optional[float]    # None -> no electrostatics
    dh_prefactor: float              # COULOMB / dielectric
    chain_ids: np.ndarray            # (N,) int
    box_length: Optional[float]      # None -> non-periodic
    positions: np.ndarray            # (N, 3) initial (unwrapped)
    residues: Optional[str] = None   # concatenated residue letters (protein)
    spec: Optional[MaterialSpec] = None
    dt: float = 0.01
    friction: float = 0.1
    nonbonded: bool = True           # False -> bonds only (ideal chain; used for validation)

    @property
    def n(self) -> int:
        return len(self.masses)

    @property
    def temperature(self) -> float:
        """Temperature that OpenMM needs to reproduce `kT` in kJ/mol."""
        return self.kT / KB_KJ_PER_MOL_K


def debye_length_nm(ionic_strength_M: float, temperature_K: float = 298.0,
                    dielectric: float = HPS_DIELECTRIC) -> Optional[float]:
    """Debye length for a 1:1 electrolyte in water; None at zero ionic strength."""
    if ionic_strength_M <= 0:
        return None
    # 0.304 nm / sqrt(I) at 298 K, eps_r = 78.5; scale for T and eps_r
    return 0.304 / math.sqrt(ionic_strength_M) * math.sqrt(
        (dielectric * temperature_K) / (78.5 * 298.0))


def _random_walk_chain(n: int, bond: float, rng: np.random.Generator,
                       min_sep: float, start: np.ndarray) -> np.ndarray:
    """Self-avoiding-ish random walk: rejects steps that land closer than
    `min_sep` to any non-adjacent earlier bead (falls back to accepting after
    50 tries; the minimizer removes any remaining overlaps)."""
    pos = np.zeros((n, 3))
    pos[0] = start
    for i in range(1, n):
        for _ in range(50):
            v = rng.normal(size=3)
            trial = pos[i - 1] + bond * v / np.linalg.norm(v)
            if i < 3 or np.min(np.linalg.norm(pos[:i - 1] - trial, axis=1)) > min_sep:
                break
        pos[i] = trial
    return pos


def build_system(spec: MaterialSpec) -> ParticleSystem:
    spec.validate()
    rng = np.random.default_rng(spec.seed)

    if spec.model == "protein":
        seq = spec.sequence
        per_chain = len(seq)
        masses = np.array([HPS_PARAMS[a][0] for a in seq] * spec.n_chains)
        sigma = np.array([HPS_PARAMS[a][1] / 10.0 for a in seq] * spec.n_chains)
        lam = np.array([HPS_PARAMS[a][2] for a in seq] * spec.n_chains)
        qmap = {"K": 1.0, "R": 1.0, "D": -1.0, "E": -1.0, "H": spec.histidine_charge}
        charge = np.array([qmap.get(a, 0.0) for a in seq] * spec.n_chains)
        kT = KB_KJ_PER_MOL_K * spec.temperature_K
        common = dict(units="real (nm, ps, amu, kJ/mol)", length_unit="nm", kT=kT,
                      epsilon=HPS_EPSILON, bond_r0=HPS_BOND_LENGTH, bond_k=HPS_BOND_K,
                      lj_cutoff=HPS_LJ_CUTOFF, dh_cutoff=HPS_DH_CUTOFF,
                      debye_length=debye_length_nm(spec.ionic_strength_M, spec.temperature_K),
                      dh_prefactor=COULOMB_KJ_NM / HPS_DIELECTRIC,
                      residues=seq * spec.n_chains, dt=0.01, friction=0.1)
        bond = HPS_BOND_LENGTH
    else:
        per_chain = int(spec.chain_length)
        n = per_chain * spec.n_chains
        lam_val = (spec.attraction_lambda if spec.attraction_lambda is not None
                   else SOLVENT_QUALITY_LAMBDA[spec.solvent_quality])
        masses, sigma = np.ones(n), np.ones(n)
        lam, charge = np.full(n, lam_val), np.zeros(n)
        common = dict(units="reduced (LJ)", length_unit="sigma", kT=1.0, epsilon=1.0,
                      bond_r0=POLYMER_BOND_LENGTH, bond_k=POLYMER_BOND_K,
                      lj_cutoff=POLYMER_LJ_CUTOFF, dh_cutoff=0.0, debye_length=None,
                      dh_prefactor=0.0, dt=0.005, friction=1.0)
        bond = POLYMER_BOND_LENGTH

    bonds = np.array([(c * per_chain + i, c * per_chain + i + 1)
                      for c in range(spec.n_chains) for i in range(per_chain - 1)], dtype=int)
    chain_ids = np.repeat(np.arange(spec.n_chains), per_chain)

    box = spec.box_length
    if spec.n_chains > 1 and box is None:
        if spec.model == "protein":
            conc = spec.concentration_mM or 2.0
            box = (spec.n_chains / (conc * 1e-3 * AVOGADRO * 1e-24)) ** (1.0 / 3.0)
        else:
            # bead number density 0.01 sigma^-3 by default (dilute solution)
            box = (per_chain * spec.n_chains / 0.01) ** (1.0 / 3.0)
    if box is not None:
        min_box = 2.0 * max(common["lj_cutoff"], common["dh_cutoff"]) + 0.1
        if box < min_box:
            raise SpecError(f"box_length {box:.2f} smaller than twice the cutoff "
                            f"({min_box:.2f}); lower the concentration or enlarge the box")

    positions = np.zeros((per_chain * spec.n_chains, 3))
    for c in range(spec.n_chains):
        start = np.zeros(3) if box is None else rng.uniform(0, box, size=3)
        chain = _random_walk_chain(per_chain, bond, rng, min_sep=bond * 1.0, start=start)
        positions[c * per_chain:(c + 1) * per_chain] = chain

    return ParticleSystem(masses=masses, sigma=sigma, lam=lam, charge=charge, bonds=bonds,
                          chain_ids=chain_ids, box_length=box, positions=positions,
                          spec=spec, **common)


# ---------------------------------------------------------------------------
# numpy reference backend
# ---------------------------------------------------------------------------

_TWO_SIXTH = 2.0 ** (1.0 / 6.0)


class NumpyForceField:
    """Ashbaugh-Hatch + Debye-Huckel + harmonic bonds, with a Verlet neighbor
    list rebuilt whenever any bead has moved more than half the skin."""

    def __init__(self, system: ParticleSystem, skin: Optional[float] = None):
        self.s = system
        n = system.n
        self.skin = skin if skin is not None else 0.15 * system.lj_cutoff
        self.cut = max(system.lj_cutoff, system.dh_cutoff if system.debye_length else 0.0)
        i, j = np.triu_indices(n, k=1)
        excluded = set(map(tuple, system.bonds.tolist()))
        if excluded:
            keep = np.array([(a, b) not in excluded for a, b in zip(i.tolist(), j.tolist())])
            i, j = i[keep], j[keep]
        self.all_i, self.all_j = i, j
        self._ref_pos = None
        self.n_rebuilds = 0

    def _min_image(self, d: np.ndarray) -> np.ndarray:
        L = self.s.box_length
        if L is not None:
            d -= L * np.round(d / L)
        return d

    def _rebuild(self, pos: np.ndarray) -> None:
        d = self._min_image(pos[self.all_j] - pos[self.all_i])
        r2 = np.einsum("ij,ij->i", d, d)
        keep = r2 < (self.cut + self.skin) ** 2
        self.i, self.j = self.all_i[keep], self.all_j[keep]
        s = self.s
        self.sig = 0.5 * (s.sigma[self.i] + s.sigma[self.j])
        self.lam_ij = 0.5 * (s.lam[self.i] + s.lam[self.j])
        self.qq = s.charge[self.i] * s.charge[self.j]
        self._ref_pos = pos.copy()
        self.n_rebuilds += 1

    def compute(self, pos: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
        s = self.s
        if s.nonbonded and (self._ref_pos is None or np.max(
                np.sum((pos - self._ref_pos) ** 2, axis=1)) > (0.5 * self.skin) ** 2):
            self._rebuild(pos)
        n = s.n
        forces = np.zeros((n, 3))
        energies = {"bond": 0.0, "ashbaugh_hatch": 0.0, "debye_huckel": 0.0}

        # bonds
        if len(s.bonds):
            a, b = s.bonds[:, 0], s.bonds[:, 1]
            d = self._min_image(pos[b] - pos[a])
            r = np.linalg.norm(d, axis=1)
            dr = r - s.bond_r0
            energies["bond"] = float(0.5 * s.bond_k * np.sum(dr ** 2))
            fvec = (-s.bond_k * dr / r)[:, None] * d      # force on b
            for k in range(3):
                forces[:, k] += np.bincount(b, fvec[:, k], minlength=n)
                forces[:, k] -= np.bincount(a, fvec[:, k], minlength=n)

        if not s.nonbonded or len(self.i) == 0:
            return forces, energies
        d = self._min_image(pos[self.j] - pos[self.i])
        r2 = np.einsum("ij,ij->i", d, d)
        r = np.sqrt(r2)
        fscal = np.zeros_like(r)                          # -dU/dr

        # Ashbaugh-Hatch
        m = r < s.lj_cutoff
        if np.any(m):
            sr6 = (self.sig[m] ** 2 / r2[m]) ** 3
            ulj = 4.0 * s.epsilon * (sr6 * sr6 - sr6)
            flj = 24.0 * s.epsilon * (2.0 * sr6 * sr6 - sr6) / r[m]
            lam = self.lam_ij[m]
            short = r[m] < _TWO_SIXTH * self.sig[m]
            u = np.where(short, ulj + (1.0 - lam) * s.epsilon, lam * ulj)
            fscal[m] += np.where(short, flj, lam * flj)
            energies["ashbaugh_hatch"] = float(np.sum(u))

        # Debye-Huckel
        if s.debye_length is not None:
            m = (r < s.dh_cutoff) & (self.qq != 0)
            if np.any(m):
                rm = r[m]
                e = s.dh_prefactor * self.qq[m] * np.exp(-rm / s.debye_length) / rm
                energies["debye_huckel"] = float(np.sum(e))
                fscal[m] += e * (1.0 / rm + 1.0 / s.debye_length)

        fvec = (fscal / r)[:, None] * d                   # force on j
        for k in range(3):
            forces[:, k] += np.bincount(self.j, fvec[:, k], minlength=n)
            forces[:, k] -= np.bincount(self.i, fvec[:, k], minlength=n)
        return forces, energies


def minimize_numpy(ff: NumpyForceField, pos: np.ndarray, max_iter: int = 2000,
                   max_disp: float = 0.05, ftol: Optional[float] = None) -> np.ndarray:
    """Adaptive steepest descent with a capped per-bead displacement."""
    pos = pos.copy()
    f, e = ff.compute(pos)
    etot = sum(e.values())
    h = max_disp * ff.s.bond_r0
    ftol = ftol if ftol is not None else 10.0 * ff.s.kT / ff.s.bond_r0
    for _ in range(max_iter):
        fmax = np.max(np.linalg.norm(f, axis=1))
        if fmax < ftol:
            break
        trial = pos + h * f / fmax
        f2, e2 = ff.compute(trial)
        et2 = sum(e2.values())
        if et2 < etot:
            pos, f, etot = trial, f2, et2
            h = min(h * 1.2, max_disp * ff.s.bond_r0 * 4)
        else:
            h *= 0.5
            if h < 1e-8:
                break
    return pos


@dataclass
class Trajectory:
    frames: np.ndarray                 # (F, N, 3) unwrapped positions
    times: np.ndarray                  # (F,) in model time units
    potential_energy: np.ndarray       # (F,)
    kinetic_temperature: np.ndarray    # (F,) instantaneous, in units of target kT
    final_positions: np.ndarray
    final_velocities: np.ndarray
    backend: str
    wall_seconds: float
    n_steps: int
    dt: float
    meta: Dict[str, Any] = field(default_factory=dict)


def _maxwell_velocities(system: ParticleSystem, rng: np.random.Generator) -> np.ndarray:
    v = rng.normal(size=(system.n, 3)) * np.sqrt(system.kT / system.masses)[:, None]
    return v - np.average(v, axis=0, weights=system.masses)


def run_numpy(system: ParticleSystem, n_steps: int, report_interval: int, seed: int = 0,
              positions: Optional[np.ndarray] = None, velocities: Optional[np.ndarray] = None,
              minimize: bool = True) -> Trajectory:
    rng = np.random.default_rng(seed)
    ff = NumpyForceField(system)
    t0 = time.time()
    x = system.positions.copy() if positions is None else positions.copy()
    if minimize:
        x = minimize_numpy(ff, x)
    v = _maxwell_velocities(system, rng) if velocities is None else velocities.copy()
    m = system.masses[:, None]
    dt, gamma = system.dt, system.friction
    c1 = math.exp(-gamma * dt)
    c2 = np.sqrt((1.0 - c1 * c1) * system.kT / m)
    f, e = ff.compute(x)
    frames, times, pe, temp = [], [], [], []
    dof = 3 * system.n
    for step in range(1, n_steps + 1):
        v += 0.5 * dt * f / m                    # B
        x += 0.5 * dt * v                        # A
        v = c1 * v + c2 * rng.normal(size=v.shape)  # O
        x += 0.5 * dt * v                        # A
        f, e = ff.compute(x)
        v += 0.5 * dt * f / m                    # B
        if step % report_interval == 0:
            frames.append(x.copy())
            times.append(step * dt)
            pe.append(sum(e.values()))
            temp.append(float(np.sum(m * v * v)) / (dof * system.kT))
    return Trajectory(frames=np.array(frames), times=np.array(times),
                      potential_energy=np.array(pe), kinetic_temperature=np.array(temp),
                      final_positions=x, final_velocities=v, backend="numpy",
                      wall_seconds=time.time() - t0, n_steps=n_steps, dt=dt,
                      meta={"neighbor_rebuilds": ff.n_rebuilds})


# ---------------------------------------------------------------------------
# OpenMM backend
# ---------------------------------------------------------------------------

def openmm_available() -> bool:
    try:
        import openmm  # noqa: F401
        return True
    except Exception:
        return False


def _build_openmm_system(system: ParticleSystem):
    try:
        import openmm
    except Exception as e:
        raise BackendNotAvailableError(
            f"openmm not importable ({type(e).__name__}: {e}); "
            "pip install openmm, or use backend='numpy'") from e

    omm = openmm.System()
    for mass in system.masses:
        omm.addParticle(float(mass))
    L = system.box_length
    if L is not None:
        omm.setDefaultPeriodicBoxVectors(openmm.Vec3(L, 0, 0), openmm.Vec3(0, L, 0),
                                         openmm.Vec3(0, 0, L))
    method = (openmm.CustomNonbondedForce.CutoffPeriodic if L is not None
              else openmm.CustomNonbondedForce.CutoffNonPeriodic)

    hb = openmm.HarmonicBondForce()
    for a, b in system.bonds.tolist():
        hb.addBond(a, b, system.bond_r0, system.bond_k)
    omm.addForce(hb)
    bond_list = [tuple(b) for b in system.bonds.tolist()]
    if not system.nonbonded:
        return omm

    ah = openmm.CustomNonbondedForce(
        "select(step(r - 2^(1/6)*s), lam*lj, lj + (1-lam)*eps);"
        "lj = 4*eps*((s/r)^12 - (s/r)^6);"
        "s = 0.5*(s1+s2); lam = 0.5*(l1+l2)")
    ah.addGlobalParameter("eps", system.epsilon)
    ah.addPerParticleParameter("s")
    ah.addPerParticleParameter("l")
    for sg, lm in zip(system.sigma, system.lam):
        ah.addParticle([float(sg), float(lm)])
    ah.setNonbondedMethod(method)
    ah.setCutoffDistance(system.lj_cutoff)
    ah.createExclusionsFromBonds(bond_list, 1)
    ah.setForceGroup(1)
    omm.addForce(ah)

    if system.debye_length is not None and np.any(system.charge != 0):
        dh = openmm.CustomNonbondedForce("A*q1*q2*exp(-r/D)/r")
        dh.addGlobalParameter("A", system.dh_prefactor)
        dh.addGlobalParameter("D", system.debye_length)
        dh.addPerParticleParameter("q")
        for q in system.charge:
            dh.addParticle([float(q)])
        dh.setNonbondedMethod(method)
        dh.setCutoffDistance(system.dh_cutoff)
        dh.createExclusionsFromBonds(bond_list, 1)
        dh.setForceGroup(2)
        omm.addForce(dh)
    return omm


def openmm_energy_forces(system: ParticleSystem, pos: np.ndarray) -> Tuple[float, np.ndarray]:
    import openmm
    omm = _build_openmm_system(system)
    integ = openmm.VerletIntegrator(0.001)
    ctx = openmm.Context(omm, integ, openmm.Platform.getPlatformByName("Reference"))
    ctx.setPositions(pos)
    st = ctx.getState(getEnergy=True, getForces=True)
    e = st.getPotentialEnergy().value_in_unit(openmm.unit.kilojoule_per_mole)
    f = np.array(st.getForces(asNumpy=True).value_in_unit(
        openmm.unit.kilojoule_per_mole / openmm.unit.nanometer))
    return e, f


def run_openmm(system: ParticleSystem, n_steps: int, report_interval: int, seed: int = 0,
               positions: Optional[np.ndarray] = None, velocities: Optional[np.ndarray] = None,
               minimize: bool = True) -> Trajectory:
    import openmm
    from openmm import unit
    omm = _build_openmm_system(system)
    integ = openmm.LangevinMiddleIntegrator(system.temperature, system.friction, system.dt)
    integ.setRandomNumberSeed(seed + 1)  # 0 means "pick randomly" in OpenMM
    try:
        platform = openmm.Platform.getPlatformByName("CPU")
        # Small CG systems get slower with more threads (sync overhead beats
        # the per-step work; measured 0.63 vs 1.49 ms/step at 1 vs 4 threads
        # for a 105-bead chain), so scale threads with system size.
        import os
        props = {"Threads": str(max(1, min(os.cpu_count() or 1, system.n // 1000)))}
    except Exception:
        platform, props = openmm.Platform.getPlatformByName("Reference"), {}
    ctx = openmm.Context(omm, integ, platform, props)
    t0 = time.time()
    ctx.setPositions(system.positions if positions is None else positions)
    if minimize:
        openmm.LocalEnergyMinimizer.minimize(ctx, 1.0, 2000)
    if velocities is None:
        ctx.setVelocities(_maxwell_velocities(system, np.random.default_rng(seed)))
    else:
        ctx.setVelocities(velocities)
    frames, times, pe, temp = [], [], [], []
    dof = 3 * system.n
    done = 0
    while done < n_steps:
        chunk = min(report_interval, n_steps - done)
        integ.step(chunk)
        done += chunk
        if chunk < report_interval:
            break
        st = ctx.getState(getPositions=True, getEnergy=True, enforcePeriodicBox=False)
        frames.append(np.array(st.getPositions(asNumpy=True).value_in_unit(unit.nanometer)))
        times.append(done * system.dt)
        pe.append(st.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))
        ke = st.getKineticEnergy().value_in_unit(unit.kilojoule_per_mole)
        temp.append(2.0 * ke / (dof * system.kT))
    st = ctx.getState(getPositions=True, getVelocities=True, enforcePeriodicBox=False)
    return Trajectory(frames=np.array(frames), times=np.array(times),
                      potential_energy=np.array(pe), kinetic_temperature=np.array(temp),
                      final_positions=np.array(st.getPositions(asNumpy=True).value_in_unit(unit.nanometer)),
                      final_velocities=np.array(st.getVelocities(asNumpy=True).value_in_unit(
                          unit.nanometer / unit.picosecond)),
                      backend=f"openmm/{platform.getName()}", wall_seconds=time.time() - t0,
                      n_steps=n_steps, dt=system.dt)


def run_md(system: ParticleSystem, n_steps: int, report_interval: int = 100,
           backend: str = "auto", seed: int = 0, **kwargs) -> Trajectory:
    """Runs Langevin MD. backend: 'auto' (openmm if installed, else numpy) | 'numpy' | 'openmm'."""
    if backend == "auto":
        backend = "openmm" if openmm_available() else "numpy"
    if backend == "openmm":
        if not openmm_available():
            raise BackendNotAvailableError("openmm requested but not installed")
        return run_openmm(system, n_steps, report_interval, seed=seed, **kwargs)
    if backend == "numpy":
        return run_numpy(system, n_steps, report_interval, seed=seed, **kwargs)
    raise ValueError(f"unknown backend {backend!r}")


# ---------------------------------------------------------------------------
# Validation harness
# ---------------------------------------------------------------------------

@dataclass
class ValidationReport:
    check: str
    passed: bool
    detail: str


def _finite_difference_check(system: ParticleSystem, pos: np.ndarray, h: float = 1e-5) -> float:
    ff = NumpyForceField(system)
    f, _ = ff.compute(pos)
    rng = np.random.default_rng(1)
    errs = []
    for idx in rng.choice(system.n, size=min(8, system.n), replace=False):
        for k in range(3):
            p1, p2 = pos.copy(), pos.copy()
            p1[idx, k] += h
            p2[idx, k] -= h
            e1 = sum(NumpyForceField(system).compute(p1)[1].values())
            e2 = sum(NumpyForceField(system).compute(p2)[1].values())
            fd = -(e1 - e2) / (2 * h)
            errs.append(abs(fd - f[idx, k]) / max(1.0, abs(f[idx, k])))
    return max(errs)


def validate_engine(include_openmm: bool = True) -> List[ValidationReport]:
    reports = []
    # A charged, mixed-hydropathy test chain exercises every term.
    test_seq = "MKEEGGSYAAAAAADEKRFWGGLPRSE"

    # 1. finite-difference forces (protein HPS, non-periodic + periodic multi-chain)
    for label, spec in [
        ("HPS single chain", MaterialSpec(model="protein", sequence=test_seq)),
        ("HPS 3 chains, periodic", MaterialSpec(model="protein", sequence=test_seq, n_chains=3,
                                                box_length=8.0)),
    ]:
        sysm = build_system(spec)
        pos = minimize_numpy(NumpyForceField(sysm), sysm.positions)
        err = _finite_difference_check(sysm, pos)
        reports.append(ValidationReport(f"finite-difference forces ({label})", err < 1e-4,
                                        f"max relative error {err:.2e}"))

    # 2. numpy vs OpenMM energies/forces
    if include_openmm and openmm_available():
        for label, spec in [
            ("HPS single chain", MaterialSpec(model="protein", sequence=test_seq)),
            ("HPS 3 chains, periodic", MaterialSpec(model="protein", sequence=test_seq,
                                                    n_chains=3, box_length=8.0)),
            ("poor-solvent polymer", MaterialSpec(model="polymer", chain_length=40,
                                                  solvent_quality="poor")),
        ]:
            sysm = build_system(spec)
            pos = minimize_numpy(NumpyForceField(sysm), sysm.positions)
            f_np, e_np = NumpyForceField(sysm).compute(pos)
            e_omm, f_omm = openmm_energy_forces(sysm, pos)
            e_err = abs(sum(e_np.values()) - e_omm) / max(1.0, abs(e_omm))
            f_err = np.max(np.abs(f_np - f_omm)) / max(1.0, np.max(np.abs(f_omm)))
            reports.append(ValidationReport(
                f"numpy vs OpenMM agreement ({label})", e_err < 1e-5 and f_err < 1e-4,
                f"energy rel err {e_err:.1e}, force rel err {f_err:.1e}"))
    else:
        reports.append(ValidationReport("numpy vs OpenMM agreement", True,
                                        "SKIPPED (openmm not installed)"))

    # 3. ideal chain (bonds only): <Ree^2> = (N-1) <b^2>, and equipartition.
    #    20 independent chains, because Ree^2 has ~100% relative fluctuation
    #    and a single chain over a short run is too noisy to test against.
    from md_analysis import end_to_end_distances  # local import: avoid a cycle
    spec = MaterialSpec(model="polymer", chain_length=20, n_chains=20)
    sysm = build_system(spec)
    sysm.nonbonded = False
    b2 = sysm.bond_r0 ** 2 + 3 * sysm.kT / sysm.bond_k       # harmonic-spring <b^2>, to O(kT/k)
    expect = (spec.chain_length - 1) * b2
    backends = ["numpy"] + (["openmm"] if include_openmm and openmm_available() else [])
    for be in backends:
        traj = run_md(sysm, n_steps=20000, report_interval=50, backend=be, seed=3, minimize=False)
        burn = len(traj.frames) // 5
        ree2 = np.mean(end_to_end_distances(traj.frames[burn:], sysm.chain_ids) ** 2)
        rel = abs(ree2 - expect) / expect
        reports.append(ValidationReport(f"ideal chain <Ree^2> = (N-1)<b^2> ({be})", rel < 0.08,
                                        f"measured {ree2:.2f}, expected {expect:.2f} (rel diff {rel:.1%})"))
        tmean = float(np.mean(traj.kinetic_temperature[burn:]))
        reports.append(ValidationReport(f"equipartition ({be})", abs(tmean - 1) < 0.03,
                                        f"<T_kin>/T_target = {tmean:.3f}"))

    # 4. solvent quality: poor-solvent chain must be more compact than good-solvent
    from md_analysis import radius_of_gyration
    rg = {}
    for q in ("good", "poor"):
        sysm = build_system(MaterialSpec(model="polymer", chain_length=40, solvent_quality=q, seed=5))
        traj = run_md(sysm, n_steps=40000, report_interval=100, seed=5)
        rg[q] = float(np.mean(radius_of_gyration(traj.frames[len(traj.frames) // 2:], sysm.chain_ids)))
    reports.append(ValidationReport("poor solvent collapses chain (Rg_poor < 0.7 Rg_good)",
                                    rg["poor"] < 0.7 * rg["good"],
                                    f"Rg good = {rg['good']:.2f}, poor = {rg['poor']:.2f} sigma"))

    # 5. HPS electrostatics: poly-Glu is more expanded at low salt than high salt
    rg = {}
    for I in (0.005, 1.0):
        sysm = build_system(MaterialSpec(model="protein", sequence="E" * 50, ionic_strength_M=I, seed=2))
        traj = run_md(sysm, n_steps=100000, report_interval=200, seed=2)
        rg[I] = float(np.mean(radius_of_gyration(traj.frames[len(traj.frames) // 2:], sysm.chain_ids)))
    reports.append(ValidationReport("salt screening compacts poly-Glu (Rg_5mM > Rg_1M)",
                                    rg[0.005] > rg[1.0],
                                    f"Rg 5 mM = {rg[0.005]:.2f} nm, 1 M = {rg[1.0]:.2f} nm"))
    return reports


if __name__ == "__main__":
    print(f"openmm available: {openmm_available()}\n")
    reports = validate_engine()
    for r in reports:
        print(f"[{'PASS' if r.passed else 'FAIL'}] {r.check}: {r.detail}")
    n_fail = sum(not r.passed for r in reports)
    print(f"\n{len(reports) - n_fail}/{len(reports)} checks passed")
    raise SystemExit(1 if n_fail else 0)
