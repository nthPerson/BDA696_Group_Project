"""Config-driven rules engine (docs/02 §4.5): ``rules.yaml`` → :class:`RuleEngine`.

A rule fires when *any* of its ``when`` conditions holds on the rep's metrics
(``RepMetrics.metrics``, keys documented in :func:`formcoach.app.pipeline.compute_rep_metrics`).
Missing or NaN metrics never fire. Every fault carries ``code, severity, metric, value,
threshold, message`` so the overlay, the event log and the report can all use it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import yaml

DEFAULT_RULES = Path(__file__).with_name("rules.yaml")
OPS = {
    "gt": lambda v, t: v > t,
    "ge": lambda v, t: v >= t,
    "lt": lambda v, t: v < t,
    "le": lambda v, t: v <= t,
    "abs_gt": lambda v, t: abs(v) > t,
}


@dataclass(frozen=True)
class Condition:
    metric: str
    op: str
    threshold: float


@dataclass(frozen=True)
class Rule:
    code: str
    severity: str
    message: str
    when: tuple[Condition, ...]
    requires: str | None = None


@dataclass(frozen=True)
class Fault:
    code: str
    severity: str
    metric: str
    value: float
    threshold: float
    op: str
    message: str

    def as_dict(self) -> dict:
        return {
            "code": self.code,
            "severity": self.severity,
            "metric": self.metric,
            "value": self.value,
            "threshold": self.threshold,
            "op": self.op,
            "message": self.message,
        }


class RuleEngine:
    def __init__(self, doc: dict):
        self.doc = doc
        self.version = doc.get("version", 1)
        self.gate: dict = doc.get("gate", {})
        self.segmentation: dict = doc.get("rep_segmentation", {})
        self._rules: dict[str, list[Rule]] = {}
        for ex, spec in doc.get("exercises", {}).items():
            rules = []
            for r in spec.get("rules", []):
                conds = tuple(
                    Condition(c["metric"], c["op"], float(c["threshold"])) for c in r["when"]
                )
                for c in conds:
                    if c.op not in OPS:
                        raise ValueError(f"{r['code']}: unknown op {c.op!r}")
                rules.append(
                    Rule(
                        r["code"], r.get("severity", "warn"), r["message"], conds, r.get("requires")
                    )
                )
            self._rules[ex] = rules

    @classmethod
    def from_yaml(cls, path: Path | None = None) -> RuleEngine:
        p = Path(path) if path else DEFAULT_RULES
        return cls(yaml.safe_load(p.read_text(encoding="utf-8")))

    @property
    def exercises(self) -> list[str]:
        return list(self._rules)

    def rules_for(self, exercise: str) -> list[Rule]:
        if exercise not in self._rules:
            raise KeyError(f"no rules for {exercise!r}; known {self.exercises}")
        return self._rules[exercise]

    def rep_config(self, exercise: str) -> dict:
        seg = self.segmentation
        cfg = dict(seg.get(exercise, {}))
        cfg.setdefault("adaptive", seg.get("adaptive", True))
        cfg.setdefault("window_s", seg.get("window_s", 6.0))
        cfg.setdefault("min_range", seg.get("min_range_deg", 25.0))
        cfg.setdefault("frac", seg.get("frac", 0.3))
        return cfg

    def evaluate(self, rep, context: set[str] | None = None) -> list[Fault]:
        """Faults for a ``RepMetrics`` (anything with ``.exercise`` and ``.metrics``). ``context``
        lists satisfied ``requires`` tags (e.g. ``{"two_arm", "frontal_view"}``); rules with an
        unmet requirement are skipped only when a context is given."""
        out: list[Fault] = []
        m = rep.metrics
        for rule in self.rules_for(rep.exercise):
            if context is not None and rule.requires and rule.requires not in context:
                continue
            for c in rule.when:
                v = m.get(c.metric)
                if v is None or (isinstance(v, float) and math.isnan(v)):
                    continue
                if OPS[c.op](float(v), c.threshold):
                    out.append(
                        Fault(
                            rule.code,
                            rule.severity,
                            c.metric,
                            float(v),
                            c.threshold,
                            c.op,
                            rule.message,
                        )
                    )
                    break
        return out

    def to_doc(self) -> dict:
        return self.doc
