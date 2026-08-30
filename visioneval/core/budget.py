"""Adaptive evaluation budget analyzer.

CPU-only, no model inference, no network. Rank catalog samples by a
deterministic risk score, then recommend how many to run.

Risk score (factors are 0 or 1; weights are configurable, these defaults)::

    risk_score = (previous_failure * 0.5)
               + (high_risk_tag * 0.2)
               + (low_confidence * 0.2)
               + (novelty * 0.1)

Sources (reuse Phase 1 attention concepts):

- previous_failure: classification SQLite ``sample_outcomes`` (same recovery
  rule as the attention sampler) plus open living-trap ``sample_id`` values.
- high_risk_tag: intersection of manifest tags and suite ``high_risk_tags``.
- low_confidence: catalog confidence at or below ``low_confidence_threshold``.
- novelty: sample has no ``sample_outcomes`` row (never evaluated). This is
  the random-coverage / novelty bucket for samples the harness has not seen.

Budget policy (deterministic, configurable):

1. Include every previous-failure sample.
2. Fill remaining slots from highest risk (ties: ``sample_id`` ascending)
   until a coverage floor is met: every configured high-risk tag that
   appears in the catalog is represented, and a small novelty/random slice
   is included (default fraction matches ``random_coverage_fraction`` 0.15).
3. ``max_budget`` (default: suite ``attention.budget``) caps the
   recommendation *after* previous failures, which always fit.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import yaml

from visioneval.core.suite import AttentionConfig, load_suite
from visioneval.core.types import ClassificationSample

_FACTOR_NAMES = ("previous_failure", "high_risk_tag", "low_confidence", "novelty")


@dataclass(frozen=True)
class RiskWeights:
    """Per-factor weights. Defaults match the documented risk formula."""

    previous_failure: float = 0.5
    high_risk_tag: float = 0.2
    low_confidence: float = 0.2
    novelty: float = 0.1

    def __post_init__(self) -> None:
        for name in _FACTOR_NAMES:
            value = getattr(self, name)
            if value < 0:
                raise ValueError(f"{name} weight must be >= 0")


@dataclass(frozen=True)
class BudgetPolicy:
    """Coverage-floor policy for the recommended evaluation budget."""

    include_all_previous_failures: bool = True
    require_high_risk_tag_coverage: bool = True
    novelty_fraction: float = 0.15
    min_novelty_count: int = 1
    max_budget: int | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.novelty_fraction <= 1.0:
            raise ValueError("novelty_fraction must be in [0, 1]")
        if self.min_novelty_count < 0:
            raise ValueError("min_novelty_count must be >= 0")
        if self.max_budget is not None and self.max_budget < 0:
            raise ValueError("max_budget must be >= 0 or None")

    @classmethod
    def from_attention(cls, attention: AttentionConfig) -> "BudgetPolicy":
        """Reuse suite attention knobs for the novelty slice and cap."""
        return cls(
            novelty_fraction=attention.random_coverage_fraction,
            max_budget=attention.budget,
        )


@dataclass(frozen=True)
class RiskFactors:
    """Binary (0 or 1) contributions. Values in [0, 1] are allowed."""

    previous_failure: float
    high_risk_tag: float
    low_confidence: float
    novelty: float

    def __post_init__(self) -> None:
        for name in _FACTOR_NAMES:
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} factor must be in [0, 1]")

    def active_labels(self) -> tuple[str, ...]:
        return tuple(name for name in _FACTOR_NAMES if getattr(self, name) > 0)


@dataclass(frozen=True)
class ScoredSample:
    sample_id: str
    label: str
    confidence: float
    tags: tuple[str, ...]
    factors: RiskFactors
    risk_score: float

    @property
    def previous_failure(self) -> bool:
        return self.factors.previous_failure > 0

    @property
    def high_risk(self) -> bool:
        return self.factors.high_risk_tag > 0

    @property
    def low_confidence(self) -> bool:
        return self.factors.low_confidence > 0

    @property
    def novelty(self) -> bool:
        return self.factors.novelty > 0

    def as_dict(self) -> dict[str, object]:
        return {
            "sample_id": self.sample_id,
            "label": self.label,
            "confidence": self.confidence,
            "tags": list(self.tags),
            "risk_score": self.risk_score,
            "factors": asdict(self.factors),
            "reasons": list(self.factors.active_labels()),
        }


@dataclass(frozen=True)
class BudgetAnalysis:
    suite_name: str
    samples: tuple[ScoredSample, ...]
    recommended_sample_ids: tuple[str, ...]
    weights: RiskWeights
    policy: BudgetPolicy
    top_n: int = 10

    @property
    def total_sample_count(self) -> int:
        return len(self.samples)

    @property
    def previous_failure_count(self) -> int:
        return sum(1 for item in self.samples if item.previous_failure)

    @property
    def high_risk_count(self) -> int:
        return sum(1 for item in self.samples if item.high_risk)

    @property
    def low_confidence_count(self) -> int:
        return sum(1 for item in self.samples if item.low_confidence)

    @property
    def recommended_budget(self) -> int:
        return len(self.recommended_sample_ids)

    @property
    def estimated_reduction_pct(self) -> float:
        return reduction_pct(self.total_sample_count, self.recommended_budget)

    def ranked_samples(self) -> tuple[ScoredSample, ...]:
        return rank_samples(self.samples)

    def top_risk_samples(self) -> tuple[ScoredSample, ...]:
        return self.ranked_samples()[: max(self.top_n, 0)]

    def as_dict(self) -> dict[str, object]:
        return {
            "suite": self.suite_name,
            "total_sample_count": self.total_sample_count,
            "previous_failure_count": self.previous_failure_count,
            "high_risk_count": self.high_risk_count,
            "low_confidence_count": self.low_confidence_count,
            "recommended_budget": self.recommended_budget,
            "estimated_reduction_pct": self.estimated_reduction_pct,
            "recommended_sample_ids": list(self.recommended_sample_ids),
            "top_risk_samples": [item.as_dict() for item in self.top_risk_samples()],
            "weights": asdict(self.weights),
            "policy": {
                "include_all_previous_failures": self.policy.include_all_previous_failures,
                "require_high_risk_tag_coverage": self.policy.require_high_risk_tag_coverage,
                "novelty_fraction": self.policy.novelty_fraction,
                "min_novelty_count": self.policy.min_novelty_count,
                "max_budget": self.policy.max_budget,
            },
        }


def score_sample(factors: RiskFactors, weights: RiskWeights | None = None) -> float:
    """Pure risk score. Factors are 0/1 (or [0, 1]); weights default as documented."""
    active = weights or RiskWeights()
    return round(
        factors.previous_failure * active.previous_failure
        + factors.high_risk_tag * active.high_risk_tag
        + factors.low_confidence * active.low_confidence
        + factors.novelty * active.novelty,
        10,
    )


def factors_for_sample(
    sample: ClassificationSample,
    *,
    high_risk_tags: Iterable[str],
    low_confidence_threshold: float,
    previous_failure_ids: Iterable[str],
    known_sample_ids: Iterable[str],
    low_confidence_enabled: bool = True,
    previous_failures_enabled: bool = True,
) -> RiskFactors:
    """Map a catalog sample plus memory onto binary risk factors."""
    previous_ids = frozenset(previous_failure_ids)
    known_ids = frozenset(known_sample_ids)
    risk_tags = frozenset(high_risk_tags)
    previous = 1.0 if previous_failures_enabled and sample.sample_id in previous_ids else 0.0
    high_risk = 1.0 if bool(sample.tags & risk_tags) else 0.0
    low_conf = (
        1.0 if low_confidence_enabled and sample.confidence <= low_confidence_threshold else 0.0
    )
    novel = 1.0 if sample.sample_id not in known_ids else 0.0
    return RiskFactors(previous, high_risk, low_conf, novel)


def score_catalog(
    samples: Sequence[ClassificationSample],
    *,
    high_risk_tags: Iterable[str],
    low_confidence_threshold: float,
    previous_failure_ids: Iterable[str],
    known_sample_ids: Iterable[str],
    weights: RiskWeights | None = None,
    low_confidence_enabled: bool = True,
    previous_failures_enabled: bool = True,
) -> tuple[ScoredSample, ...]:
    """Score every sample. Input order does not affect scores."""
    active = weights or RiskWeights()
    scored: list[ScoredSample] = []
    for sample in samples:
        factors = factors_for_sample(
            sample,
            high_risk_tags=high_risk_tags,
            low_confidence_threshold=low_confidence_threshold,
            previous_failure_ids=previous_failure_ids,
            known_sample_ids=known_sample_ids,
            low_confidence_enabled=low_confidence_enabled,
            previous_failures_enabled=previous_failures_enabled,
        )
        scored.append(
            ScoredSample(
                sample_id=sample.sample_id,
                label=sample.label,
                confidence=sample.confidence,
                tags=tuple(sorted(sample.tags)),
                factors=factors,
                risk_score=score_sample(factors, active),
            )
        )
    return tuple(scored)


def rank_samples(scored: Sequence[ScoredSample]) -> tuple[ScoredSample, ...]:
    """Deterministic order: higher score first, then sample_id ascending."""
    return tuple(sorted(scored, key=lambda item: (-item.risk_score, item.sample_id)))


def novelty_floor(total: int, available_novel: int, policy: BudgetPolicy) -> int:
    """How many novel samples the coverage floor requires."""
    if total <= 0 or available_novel <= 0:
        return 0
    wanted = max(policy.min_novelty_count, math.ceil(policy.novelty_fraction * total))
    return min(available_novel, wanted)


def recommend_budget(
    scored: Sequence[ScoredSample],
    policy: BudgetPolicy | None = None,
    high_risk_tags: Iterable[str] = (),
) -> tuple[str, ...]:
    """Select sample ids to run under the documented coverage-floor policy."""
    active = policy or BudgetPolicy()
    ranked = rank_samples(scored)
    selected: list[ScoredSample] = []
    selected_ids: set[str] = set()

    def add(item: ScoredSample) -> None:
        if item.sample_id not in selected_ids:
            selected.append(item)
            selected_ids.add(item.sample_id)

    if active.include_all_previous_failures:
        for item in ranked:
            if item.previous_failure:
                add(item)

    present_tags = {tag for item in scored for tag in item.tags}
    required_tags = frozenset(high_risk_tags) & present_tags
    needed_novelty = novelty_floor(
        len(scored), sum(1 for item in scored if item.novelty), active
    )

    def floor_met() -> bool:
        tags_ok = True
        if active.require_high_risk_tag_coverage and required_tags:
            covered = {tag for item in selected for tag in item.tags if tag in required_tags}
            tags_ok = required_tags <= covered
        novelty_ok = sum(1 for item in selected if item.novelty) >= needed_novelty
        return tags_ok and novelty_ok

    remaining = [item for item in ranked if item.sample_id not in selected_ids]
    for item in remaining:
        if floor_met():
            break
        if active.max_budget is not None and len(selected) >= active.max_budget:
            break
        add(item)

    if not selected and ranked:
        add(ranked[0])

    return tuple(item.sample_id for item in selected)


def reduction_pct(total: int, recommended: int) -> float:
    if total <= 0:
        return 0.0
    return round(100.0 * (1.0 - recommended / total), 1)


def analyze_samples(
    samples: Sequence[ClassificationSample],
    *,
    suite_name: str,
    high_risk_tags: Iterable[str],
    low_confidence_threshold: float,
    previous_failure_ids: Iterable[str],
    known_sample_ids: Iterable[str],
    weights: RiskWeights | None = None,
    policy: BudgetPolicy | None = None,
    low_confidence_enabled: bool = True,
    previous_failures_enabled: bool = True,
    top_n: int = 10,
) -> BudgetAnalysis:
    """Pure analysis over an in-memory catalog."""
    active_weights = weights or RiskWeights()
    active_policy = policy or BudgetPolicy()
    scored = score_catalog(
        samples,
        high_risk_tags=high_risk_tags,
        low_confidence_threshold=low_confidence_threshold,
        previous_failure_ids=previous_failure_ids,
        known_sample_ids=known_sample_ids,
        weights=active_weights,
        low_confidence_enabled=low_confidence_enabled,
        previous_failures_enabled=previous_failures_enabled,
    )
    recommended = recommend_budget(scored, active_policy, high_risk_tags)
    return BudgetAnalysis(
        suite_name=suite_name,
        samples=scored,
        recommended_sample_ids=recommended,
        weights=active_weights,
        policy=active_policy,
        top_n=top_n,
    )


def analyze_suite(
    suite_path: Path,
    *,
    weights: RiskWeights | None = None,
    policy: BudgetPolicy | None = None,
    traps_db: Path | None = None,
    top_n: int = 10,
) -> BudgetAnalysis:
    """Load a Phase 1 suite YAML plus local SQLite memory. No inference."""
    path = suite_path.expanduser()
    suite = load_suite(path)
    manifest_path = resolve_suite_relative(path, suite.dataset.manifest)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"dataset manifest not found: {manifest_path}")
    samples = load_manifest_samples(manifest_path)
    cache_path = resolve_suite_relative(path, suite.cache.path)
    previous_ids, known_ids = load_failure_memory(cache_path)
    trap_ids = load_open_trap_sample_ids(traps_db)
    previous_ids = previous_ids | trap_ids
    active_policy = policy or BudgetPolicy.from_attention(suite.attention)
    return analyze_samples(
        samples,
        suite_name=suite.name,
        high_risk_tags=suite.attention.high_risk_tags,
        low_confidence_threshold=suite.attention.low_confidence_threshold,
        previous_failure_ids=previous_ids,
        known_sample_ids=known_ids,
        weights=weights,
        policy=active_policy,
        low_confidence_enabled=suite.attention.low_confidence,
        previous_failures_enabled=suite.attention.previous_failures,
        top_n=top_n,
    )


def load_manifest_samples(path: Path) -> list[ClassificationSample]:
    """Read a classification manifest without touching the runner or adapters."""
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "samples" not in payload:
        raise ValueError("manifest YAML must contain a top-level 'samples' list")
    samples: list[ClassificationSample] = []
    seen: set[str] = set()
    for raw in payload["samples"]:
        if not isinstance(raw, dict):
            raise ValueError("each manifest sample must be a mapping")
        sample_id = str(raw.get("id") or raw.get("sample_id") or "")
        if not sample_id:
            raise ValueError("manifest sample is missing id")
        if sample_id in seen:
            raise ValueError(f"duplicate sample_id: {sample_id}")
        seen.add(sample_id)
        label = str(raw.get("label") or "")
        confidence = float(raw.get("confidence", 1.0))
        tags = frozenset(str(tag) for tag in (raw.get("tags") or []))
        image_path = raw.get("image_path")
        samples.append(
            ClassificationSample(
                sample_id,
                label,
                confidence,
                tags,
                False,
                str(image_path) if image_path else None,
            )
        )
    return samples


def resolve_suite_relative(suite_path: Path, relative: str) -> Path:
    """Resolve suite paths next to the YAML, then cwd (matches CLI usage)."""
    raw = Path(relative)
    if raw.is_absolute():
        return raw
    suite_dir = suite_path.parent
    candidate = (suite_dir / raw)
    if candidate.exists():
        return candidate
    cwd_candidate = Path.cwd() / raw
    if cwd_candidate.exists():
        return cwd_candidate
    return candidate


def load_failure_memory(cache_path: Path) -> tuple[set[str], set[str]]:
    """Read previous-failure and known ids. Missing files are empty memory."""
    if not cache_path.is_file():
        return set(), set()
    from visioneval.core.cache import SQLiteCache

    cache = SQLiteCache(cache_path)
    return cache.previous_failure_ids(), cache.known_sample_ids()


def load_open_trap_sample_ids(traps_db: Path | None) -> set[str]:
    """Open living traps count as previous failures when the DB exists."""
    if traps_db is None or not traps_db.is_file():
        return set()
    from visioneval.traps.store import TrapStore

    return {trap.sample_id for trap in TrapStore(traps_db).list_open() if trap.sample_id}


def format_human(analysis: BudgetAnalysis) -> str:
    """Human-readable report for the CLI (no colour, deterministic)."""
    lines = [
        f"VisionEval budget analysis: {analysis.suite_name}",
        "",
        f"Total samples:         {analysis.total_sample_count}",
        f"Previous failures:     {analysis.previous_failure_count}",
        f"High-risk:             {analysis.high_risk_count}",
        f"Low-confidence:        {analysis.low_confidence_count}",
        f"Recommended budget:    {analysis.recommended_budget}",
        f"Estimated reduction:   {analysis.estimated_reduction_pct:.1f}%",
        "",
        "Top-risk samples:",
    ]
    top = analysis.top_risk_samples()
    if not top:
        lines.append("  (none)")
    else:
        width = max(len(item.sample_id) for item in top)
        for item in top:
            reasons = ",".join(item.factors.active_labels()) or "none"
            marker = "*" if item.sample_id in analysis.recommended_sample_ids else " "
            lines.append(
                f"  {marker} {item.sample_id:<{width}}  {item.risk_score:.3f}  {reasons}"
            )
    lines.extend(
        [
            "",
            "Recommended sample ids: "
            + (", ".join(analysis.recommended_sample_ids) or "(none)"),
            "",
            "Policy: include all previous failures, then fill from highest risk",
            "until high-risk tags are represented and a novelty/random slice is",
            f"included (fraction={analysis.policy.novelty_fraction}, "
            f"min={analysis.policy.min_novelty_count}, "
            f"max_budget={analysis.policy.max_budget}).",
            "Weights: "
            f"previous_failure={analysis.weights.previous_failure} "
            f"high_risk_tag={analysis.weights.high_risk_tag} "
            f"low_confidence={analysis.weights.low_confidence} "
            f"novelty={analysis.weights.novelty}",
        ]
    )
    return "\n".join(lines) + "\n"


def to_json(analysis: BudgetAnalysis) -> str:
    return json.dumps(analysis.as_dict(), indent=2, sort_keys=True) + "\n"


def default_traps_db() -> Path:
    return Path("artifacts/traps.sqlite3")
