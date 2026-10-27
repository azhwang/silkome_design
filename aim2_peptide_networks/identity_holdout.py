"""
Identity-Based Holdout Partitioning for Aim 2
================================================

The proposal is specific about this:

    "on a holdout constructed by sequence identity rather than by random
     partition"

That requirement is load-bearing. Self-assembling peptide libraries are
dense with near-duplicates — single-residue variants, the same core with a
different terminal cap, scrambled controls. Under a random split, a test
peptide's near-twin sits in the training set, so a sequence-embedding
baseline can score well by recall rather than by prediction, and a composed
chain gets evaluated against an inflated bar. Worse, the inflation is
invisible: nothing in the metrics says the split leaked.

This module builds a partition with an explicit, verified guarantee:

    no held-out peptide exceeds `threshold` identity to ANY training peptide

It gets there by single-linkage clustering at the identity threshold and
assigning whole clusters, which makes the guarantee hold by construction —
and then checks it empirically anyway, because "by construction" is exactly
the kind of claim that quietly stops being true after a refactor.

Identity definition
---------------------
Peptide gelator libraries are short (dipeptides to ~20-mers) and include
insertions and deletions, so a fixed-column identity would be wrong. We use

    identity(a, b) = 1 - levenshtein(a, b) / max(len(a), len(b))

which reduces to ordinary percent identity for equal-length sequences and
degrades sensibly across length differences.

Terminal modifications are handled conservatively: Fmoc-FF and FF are
chemically distinct molecules (and assemble differently), so they are
separate entries — but because they share a core, they are forced into the
same cluster rather than being allowed to straddle the split. Conservative
in the direction that prevents leakage.

Cost
------
All-pairs Levenshtein over a few thousand peptides is 10^6-10^7 cell
updates in pure Python, which is slow enough to matter. A sound composition
bound prunes almost all of it before any DP runs:

    levenshtein(a, b) >= max(|a|, |b|) - |multiset intersection of a, b|

because an alignment can only match equal characters, so the number of
matched positions is at most the multiset intersection, and every unmatched
position in the longer sequence costs at least one edit. Therefore

    identity(a, b) <= |multiset intersection| / max(|a|, |b|)

If that upper bound is already below the threshold, the pair cannot be
linked and the DP is skipped. The bound is an upper bound, so pruning on it
never discards a pair that would have been linked — no approximation.

Runnable standalone: `python3 aim2_peptide_networks/identity_holdout.py`
"""

import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


class HoldoutError(ValueError):
    """Raised when a requested partition cannot be built as specified."""


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------

def levenshtein(a: str, b: str) -> int:
    """Standard edit distance, two-row DP.

    Written out rather than pulled from a dependency so the whole identity
    definition is inspectable in one file — the holdout guarantee rests on
    it, and per CLAUDE.md this repo prefers real dependency-free
    implementations where the logic does not need an external service.
    """
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(min(
                previous[j] + 1,          # deletion
                current[j - 1] + 1,       # insertion
                previous[j - 1] + (ca != cb),  # substitution
            ))
        previous = current
    return previous[-1]


def sequence_identity(a: str, b: str) -> float:
    """Identity in [0, 1]; 1.0 for identical sequences."""
    if not a and not b:
        return 1.0
    longest = max(len(a), len(b))
    return 1.0 - levenshtein(a, b) / longest


def identity_upper_bound(
    a: str,
    b: str,
    counts_a: Optional[Counter] = None,
    counts_b: Optional[Counter] = None,
) -> float:
    """Sound upper bound on `sequence_identity(a, b)` from composition alone.

    Cheap (one pass over the smaller composition) and never underestimates,
    so pruning with it cannot drop a pair that should have been linked.
    """
    if counts_a is None:
        counts_a = Counter(a)
    if counts_b is None:
        counts_b = Counter(b)
    if len(counts_a) > len(counts_b):
        counts_a, counts_b = counts_b, counts_a
    overlap = sum(min(n, counts_b.get(residue, 0)) for residue, n in counts_a.items())
    longest = max(len(a), len(b))
    if longest == 0:
        return 1.0
    return overlap / longest


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


@dataclass
class ClusterResult:
    clusters: List[List[str]]          # cluster -> member identity keys
    cluster_of: Dict[str, int]         # identity key -> cluster index
    threshold: float
    n_pairs_considered: int
    n_pairs_pruned: int
    n_links: int

    @property
    def n_clusters(self) -> int:
        return len(self.clusters)

    @property
    def largest_cluster_size(self) -> int:
        return max((len(c) for c in self.clusters), default=0)


def _core_sequence(identity_key: str) -> str:
    """Extract the bare residue sequence from an identity key.

    Accepts either a plain sequence ("FF") or the `n|SEQ|c` identity key
    emitted by `dataset_schema.PeptideNetworkRecord.identity_key`, so this
    module works both standalone and against real records.
    """
    if "|" in identity_key:
        parts = identity_key.split("|")
        if len(parts) == 3:
            return parts[1]
    return identity_key


def cluster_by_identity(
    identity_keys: Sequence[str],
    threshold: float = 0.6,
) -> ClusterResult:
    """Single-linkage cluster peptides at `threshold` identity.

    Single linkage is the correct choice here even though it chains: the
    guarantee we need is about the maximum identity across the split, and
    single linkage is precisely the clustering under which "different
    clusters" implies "no pair above threshold". Average or complete linkage
    would give tidier clusters and a weaker, violable guarantee.

    Entries sharing a core sequence but differing only in terminal
    modification are linked unconditionally.
    """
    if not 0.0 < threshold <= 1.0:
        raise HoldoutError(f"threshold must be in (0, 1], got {threshold}")

    keys = list(dict.fromkeys(identity_keys))  # de-duplicate, keep order
    if not keys:
        raise HoldoutError("no peptides to cluster")

    cores = [_core_sequence(k) for k in keys]
    counts = [Counter(c) for c in cores]

    uf = _UnionFind(len(keys))
    considered = pruned = links = 0

    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            considered += 1

            # Same core, different cap -> always linked, no distance needed.
            if cores[i] == cores[j]:
                uf.union(i, j)
                links += 1
                continue

            # Length prefilter: identity <= min_len / max_len regardless of
            # content, since at most min_len positions can ever match.
            shortest = min(len(cores[i]), len(cores[j]))
            longest = max(len(cores[i]), len(cores[j]))
            if shortest / longest < threshold:
                pruned += 1
                continue

            if identity_upper_bound(cores[i], cores[j], counts[i], counts[j]) < threshold:
                pruned += 1
                continue

            if sequence_identity(cores[i], cores[j]) >= threshold:
                uf.union(i, j)
                links += 1

    grouped: Dict[int, List[str]] = {}
    for index, key in enumerate(keys):
        grouped.setdefault(uf.find(index), []).append(key)

    clusters = sorted(grouped.values(), key=lambda c: (-len(c), c[0]))
    cluster_of = {
        key: idx for idx, members in enumerate(clusters) for key in members
    }

    return ClusterResult(
        clusters=clusters,
        cluster_of=cluster_of,
        threshold=threshold,
        n_pairs_considered=considered,
        n_pairs_pruned=pruned,
        n_links=links,
    )


# ---------------------------------------------------------------------------
# Partitioning
# ---------------------------------------------------------------------------

@dataclass
class Partition:
    train: List[str]
    test: List[str]
    threshold: float
    target_test_fraction: float
    achieved_test_fraction: float
    n_clusters: int
    n_test_clusters: int
    largest_cluster_size: int
    max_cross_identity: Optional[float] = None
    violations: List[Tuple[str, str, float]] = field(default_factory=list)

    @property
    def is_verified(self) -> bool:
        return not self.violations

    def summary(self) -> str:
        lines = [
            "Identity-based holdout partition",
            "=" * 60,
            f"identity threshold           : {self.threshold:.2f}",
            f"train peptides               : {len(self.train)}",
            f"test peptides                : {len(self.test)}",
            f"test fraction                : {self.achieved_test_fraction:.1%} "
            f"(target {self.target_test_fraction:.1%})",
            f"identity clusters            : {self.n_clusters} "
            f"({self.n_test_clusters} held out)",
            f"largest cluster              : {self.largest_cluster_size} peptides",
        ]
        if self.max_cross_identity is not None:
            lines.append(
                f"max train-test identity      : {self.max_cross_identity:.3f}"
            )
        if self.violations:
            lines.append("")
            lines.append(
                f"!! GUARANTEE VIOLATED: {len(self.violations)} train-test pairs "
                f"at or above threshold"
            )
            for a, b, ident in self.violations[:5]:
                lines.append(f"   {a} <-> {b}  identity {ident:.3f}")
        else:
            lines.append("guarantee                    : VERIFIED "
                         "(no train-test pair at or above threshold)")

        if self.largest_cluster_size > max(1, len(self.train) + len(self.test)) // 4:
            lines.append("")
            lines.append(
                "NOTE: one cluster holds over a quarter of the library. Single "
                "linkage has chained a large group, so the usable split is "
                "coarser than the peptide count suggests — report the cluster "
                "count, not the peptide count, as the effective sample size."
            )
        return "\n".join(lines)


def make_identity_holdout(
    identity_keys: Sequence[str],
    threshold: float = 0.6,
    test_fraction: float = 0.2,
    seed: int = 0,
    verify: bool = True,
) -> Partition:
    """Partition peptides so no test peptide is >= `threshold` identical to train.

    Whole clusters are assigned, so the guarantee holds by construction. The
    achieved test fraction will rarely equal the target exactly — clusters
    are indivisible, and forcing an exact fraction would mean splitting one,
    which is precisely the leak this function exists to prevent. The
    achieved fraction is reported rather than silently pursued.
    """
    if not 0.0 < test_fraction < 1.0:
        raise HoldoutError(f"test_fraction must be in (0, 1), got {test_fraction}")

    clustering = cluster_by_identity(identity_keys, threshold=threshold)
    total = sum(len(c) for c in clustering.clusters)
    target_count = test_fraction * total

    # Shuffle cluster order with a fixed seed, then fill the test side until
    # adding the next cluster would overshoot the target more than stopping
    # short would undershoot it. Deterministic given the seed.
    order = list(range(clustering.n_clusters))
    random.Random(seed).shuffle(order)

    test_clusters: List[int] = []
    test_count = 0
    for index in order:
        size = len(clustering.clusters[index])
        if test_count == 0:
            # Always take at least one cluster, else an empty test set.
            test_clusters.append(index)
            test_count += size
            continue
        if abs(test_count + size - target_count) < abs(test_count - target_count):
            test_clusters.append(index)
            test_count += size
        if test_count >= target_count:
            break

    test_set = set(test_clusters)
    test = sorted(
        key for idx in test_set for key in clustering.clusters[idx]
    )
    train = sorted(
        key for idx, members in enumerate(clustering.clusters)
        if idx not in test_set for key in members
    )

    if not train:
        raise HoldoutError(
            "every cluster landed in the test set; the library is a single "
            "identity group at this threshold and cannot be split"
        )

    partition = Partition(
        train=train,
        test=test,
        threshold=threshold,
        target_test_fraction=test_fraction,
        achieved_test_fraction=len(test) / total,
        n_clusters=clustering.n_clusters,
        n_test_clusters=len(test_clusters),
        largest_cluster_size=clustering.largest_cluster_size,
    )

    if verify:
        max_identity, violations = verify_partition(train, test, threshold)
        partition.max_cross_identity = max_identity
        partition.violations = violations

    return partition


def verify_partition(
    train: Iterable[str],
    test: Iterable[str],
    threshold: float,
) -> Tuple[Optional[float], List[Tuple[str, str, float]]]:
    """Independently check the no-leakage guarantee.

    Recomputed from scratch against the final partition rather than trusting
    the clustering that produced it. The guarantee is the entire value of an
    identity-based split, so it is worth paying for a direct check: a split
    that claims it and does not have it is worse than an honest random split,
    because the inflated score carries a correctness claim.
    """
    train_list = list(train)
    test_list = list(test)
    train_cores = [(k, _core_sequence(k)) for k in train_list]
    test_cores = [(k, _core_sequence(k)) for k in test_list]
    train_counts = [Counter(c) for _, c in train_cores]

    max_identity: Optional[float] = None
    violations: List[Tuple[str, str, float]] = []

    for test_key, test_core in test_cores:
        test_count = Counter(test_core)
        for (train_key, train_core), train_count in zip(train_cores, train_counts):
            bound = identity_upper_bound(
                test_core, train_core, test_count, train_count
            )
            if max_identity is not None and bound <= max_identity:
                # Cannot raise the running maximum and cannot be a violation
                # unless it already exceeds the threshold.
                if bound < threshold:
                    continue
            identity = sequence_identity(test_core, train_core)
            if max_identity is None or identity > max_identity:
                max_identity = identity
            if identity >= threshold:
                violations.append((test_key, train_key, identity))

    return max_identity, violations


# ---------------------------------------------------------------------------
# Demo / smoke test
# ---------------------------------------------------------------------------

def _demo_library() -> List[str]:
    """A library with deliberate near-duplicate structure.

    Built to contain the exact failure a random split would hide: families
    of single-residue variants, a capped/uncapped pair sharing a core, and
    some genuinely unrelated sequences.
    """
    return [
        # Fmoc-dipeptide family, all mutually close.
        "FMOC|FF|OH", "FMOC|FL|OH", "FMOC|FY|OH", "FMOC|FV|OH",
        # The same FF core without the cap: must not straddle the split.
        "H|FF|OH",
        # A RADA-type self-assembling family.
        "H|RADARADARADARADA|NH2", "H|RADARADARADARADARADA|NH2",
        "H|RADARADARADARAEA|NH2",
        # An amyloid-derived family.
        "H|KLVFF|OH", "H|KLVFFA|OH", "H|KLVFFAE|OH",
        # Singletons, unrelated to the above and to each other.
        "H|IKVAV|OH", "H|GHRPS|OH", "H|WWDPTT|OH", "H|CNGRC|OH",
    ]


def _run_demo() -> None:
    print(__doc__.strip().splitlines()[0])
    print()

    print("--- identity definition ---")
    checks = [
        ("FF", "FF", 1.0),
        ("FF", "FL", 0.5),
        ("KLVFF", "KLVFFA", 5 / 6),
        ("FF", "RADARADA", 0.0),
    ]
    for a, b, expected in checks:
        got = sequence_identity(a, b)
        status = "OK" if abs(got - expected) < 1e-9 else "MISMATCH"
        print(f"  identity({a!r}, {b!r}) = {got:.3f}  expected {expected:.3f}  {status}")
        assert status == "OK", f"identity({a}, {b}) = {got}, expected {expected}"

    print()
    print("--- composition bound is sound (never below true identity) ---")
    library = [_core_sequence(k) for k in _demo_library()]
    worst_slack = 1.0
    for i in range(len(library)):
        for j in range(len(library)):
            true_identity = sequence_identity(library[i], library[j])
            bound = identity_upper_bound(library[i], library[j])
            assert bound >= true_identity - 1e-12, (
                f"bound {bound} below true identity {true_identity} for "
                f"{library[i]!r}/{library[j]!r} — pruning would be unsound"
            )
            worst_slack = min(worst_slack, bound - true_identity)
    print(f"  checked {len(library) ** 2} pairs; bound never fell below the true "
          f"identity (tightest slack {worst_slack:.3f})")

    print()
    print("--- clustering ---")
    clustering = cluster_by_identity(_demo_library(), threshold=0.6)
    print(f"  {clustering.n_clusters} clusters from {len(_demo_library())} peptides "
          f"at identity >= 0.60")
    print(f"  pruned {clustering.n_pairs_pruned}/{clustering.n_pairs_considered} "
          f"pairs before any edit-distance DP")
    for members in clustering.clusters:
        if len(members) > 1:
            print(f"    [{len(members)}] {', '.join(members)}")

    # The capped and uncapped FF must share a cluster, or the split leaks.
    ff_capped = clustering.cluster_of["FMOC|FF|OH"]
    ff_bare = clustering.cluster_of["H|FF|OH"]
    assert ff_capped == ff_bare, (
        "Fmoc-FF and FF ended up in different clusters — they share a core "
        "and could straddle the split"
    )
    print("  check: Fmoc-FF and bare FF share a cluster -> OK")

    print()
    partition = make_identity_holdout(_demo_library(), threshold=0.6,
                                      test_fraction=0.3, seed=1)
    print(partition.summary())
    assert partition.is_verified, partition.violations
    assert partition.max_cross_identity is not None
    assert partition.max_cross_identity < 0.6

    print()
    print("--- a random split on the same library, for contrast ---")
    keys = _demo_library()
    shuffled = list(keys)
    random.Random(1).shuffle(shuffled)
    cut = max(1, int(0.3 * len(shuffled)))
    random_test, random_train = shuffled[:cut], shuffled[cut:]
    max_identity, violations = verify_partition(random_train, random_test, 0.6)
    print(f"  max train-test identity      : {max_identity:.3f}")
    print(f"  pairs at or above 0.60       : {len(violations)}")
    if violations:
        for a, b, ident in violations[:3]:
            print(f"    {a} <-> {b}  identity {ident:.3f}")
        print("  -> a random split leaks near-duplicates across the boundary, "
              "which is what the identity split is for.")

    print()
    print("All identity_holdout.py self-checks passed.")


if __name__ == "__main__":
    _run_demo()
