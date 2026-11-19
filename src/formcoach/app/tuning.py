"""Tuning: the parameter registry, an overrides set over a rules file, and re-judging (Bench
design §6.1).

The lab console shows one slider per :class:`Param`. Moving one never edits the packaged rules
file: it changes an :class:`Overrides` set over a base :class:`~formcoach.rules.engine.RuleEngine`.
:meth:`Overrides.engine` is the base engine with the overridden rule thresholds, used to judge
new reps, and :func:`rejudge` judges reps already counted again from their stored metrics, so
nothing re-runs pose. :meth:`Overrides.document` is the complete rules document with every
override applied (rule thresholds, the energy gate and the rep counter's minimum range), saved
with each capture; :meth:`Overrides.save` writes it as a YAML rules file for review.

Keys:

- ``rule.<exercise>.<CODE>.<metric>``: the threshold of one condition of one rule, in the
  metric's own unit (``deg``, ``s``, or torso lengths for the offset metrics, unit ``""``).
  Range ``[min(0, 2.5 * default), max(2.5 * default, default + 1)]``; step 1 for a default of
  10 or more, 0.1 for 1 or more, else 0.01.
- ``gate.energy_threshold``: the laptop energy gate's threshold on ``var(|a|)``, (m/s²)²,
  0.5-30 in steps of 0.5.
- ``gate.on_windows`` / ``gate.off_windows``: consecutive active/idle windows to open/close the
  energy gate, 1-10 and 1-40, whole numbers.
- ``seg.min_range_deg``: the angle range (degrees) the rep counter must see before it counts,
  10-60 in steps of 1.
"""

from __future__ import annotations

import copy
import math
import numbers
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml

from formcoach.io.protocol import GATE_ENERGY_THRESHOLD_MS2SQ, GATE_OFF_WINDOWS, GATE_ON_WINDOWS
from formcoach.pose.reps import ADAPTIVE_DEFAULTS
from formcoach.rules.engine import DEFAULT_RULES, RuleEngine

# the metrics (keys of RepMetrics.metrics) measured in degrees; duration_s is seconds and the
# offset metrics (elbow_drift_max, knee_track_at_bottom, hip_knee_height_at_bottom) torso lengths
DEGREE_METRICS = frozenset(
    {
        "elbow_min",
        "elbow_max",
        "elbow_min_l",
        "elbow_min_r",
        "elbow_max_l",
        "elbow_max_r",
        "elbow_asymmetry",
        "elbow_top_asymmetry",
        "upper_arm_trunk_range",
        "upper_arm_trunk_drift",
        "trunk_incl_max",
        "abd_peak",
        "elbow_at_abd_peak",
        "knee_min",
        "trunk_incl_at_bottom",
    }
)
WHOLE_NUMBER_KEYS = frozenset({"gate.on_windows", "gate.off_windows"})
PACKAGE_RULES_DIR = DEFAULT_RULES.parent


@dataclass(frozen=True)
class Param:
    """One slider: ``key`` (see the module doc), a short ``label``, a ``group`` (``"rules"``,
    ``"gate"`` or ``"counter"``), the ``unit`` (``"deg"``, ``"s"``, ``"(m/s²)²"``,
    ``"windows"`` or ``""``), the allowed ``min``/``max``, the slider ``step`` and the base
    rules file's ``default``, all in that unit."""

    key: str
    label: str
    group: str
    unit: str
    min: float
    max: float
    step: float
    default: float


@dataclass(frozen=True)
class _Setting:
    """A non-rule parameter: where it lives in the rules document and its fallback default."""

    label: str
    group: str
    unit: str
    min: float
    max: float
    step: float
    section: str
    field: str
    fallback: float


SETTINGS: dict[str, _Setting] = {
    "gate.energy_threshold": _Setting(
        "Energy threshold", "gate", "(m/s²)²", 0.5, 30.0, 0.5,
        "gate", "energy_threshold_ms2sq", GATE_ENERGY_THRESHOLD_MS2SQ,
    ),
    "gate.on_windows": _Setting(
        "Windows to open", "gate", "windows", 1.0, 10.0, 1.0, "gate", "on_windows", GATE_ON_WINDOWS
    ),
    "gate.off_windows": _Setting(
        "Windows to close", "gate", "windows", 1.0, 40.0, 1.0,
        "gate", "off_windows", GATE_OFF_WINDOWS,
    ),
    "seg.min_range_deg": _Setting(
        "Minimum range", "counter", "deg", 10.0, 60.0, 1.0,
        "rep_segmentation", "min_range_deg", ADAPTIVE_DEFAULTS["min_range"],
    ),
}  # fmt: skip


def rule_key(exercise: str, code: str, metric: str) -> str:
    """The parameter key of one rule condition: ``rule.<exercise>.<CODE>.<metric>``."""
    return f"rule.{exercise}.{code}.{metric}"


def _metric_unit(metric: str) -> str:
    if metric == "duration_s":
        return "s"
    return "deg" if metric in DEGREE_METRICS else ""


def _rule_param(exercise: str, code: str, metric: str, default: float) -> Param:
    lo = min(0.0, 2.5 * default)  # 0, unless the limit itself is negative
    hi = max(2.5 * default, default + 1.0)
    step = 1.0 if default >= 10 else 0.1 if default >= 1 else 0.01
    return Param(
        key=rule_key(exercise, code, metric),
        label=f"{code}: {metric}",
        group="rules",
        unit=_metric_unit(metric),
        min=round(lo, 6),
        max=round(hi, 6),
        step=step,
        default=float(default),
    )


class Overrides:
    """Slider values over a base rules engine; the base and its document are never changed.

    ``base_name`` names the base rules file (for the saved file's header and ``meta.json``).
    ``values`` holds only the keys whose value differs from the default.
    """

    def __init__(self, base: RuleEngine, base_name: str):
        self.base = base
        self.base_name = base_name
        self._values: dict[str, float] = {}
        self._engine: RuleEngine | None = None  # cached engine for the current rule values
        self._rule_params: dict[str, list[Param]] = {}  # exercise -> its rule params, in order
        self._params: dict[str, Param] = {}
        for ex in base.exercises:
            params = []
            for rule in base.rules_for(ex):
                for c in rule.when:
                    params.append(_rule_param(ex, rule.code, c.metric, c.threshold))
            self._rule_params[ex] = params
            self._params.update((p.key, p) for p in params)
        doc = base.doc
        for key, s in SETTINGS.items():
            default = float((doc.get(s.section) or {}).get(s.field, s.fallback))
            self._params[key] = Param(key, s.label, s.group, s.unit, s.min, s.max, s.step, default)

    @property
    def values(self) -> dict[str, float]:
        """The overridden keys and their values (a copy); keys at their default are absent."""
        return dict(self._values)

    def params(self, exercise: str) -> list[Param]:
        """The sliders for ``exercise``: one per condition of each of its rules (in the rules
        file's order), then the energy gate's three and the rep counter's minimum range.

        Raises:
            KeyError: the base rules have no ``exercise``.
        """
        if exercise not in self._rule_params:
            raise KeyError(f"no rules for {exercise!r}; known {self.base.exercises}")
        return [*self._rule_params[exercise], *(self._params[k] for k in SETTINGS)]

    def default(self, key: str) -> float:
        """The base rules file's value of ``key``.

        Raises:
            KeyError: ``key`` is not a parameter.
        """
        return self._param(key).default

    def set(self, key: str, value: float) -> float:
        """Override ``key`` with ``value`` clamped to the parameter's range (window counts are
        rounded to whole numbers); returns the value applied. Setting a key back to its default
        removes it from :attr:`values`.

        Raises:
            KeyError: ``key`` is not a parameter.
            ValueError: ``value`` is NaN or not a number. Nothing changes on either error.
        """
        p = self._param(key)
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            raise ValueError(f"{key}: expected a number, got {value!r}")
        v = float(value)
        if math.isnan(v):
            raise ValueError(f"{key}: value is NaN")
        v = min(max(v, p.min), p.max)
        if key in WHOLE_NUMBER_KEYS:
            v = float(round(v))
        if v == p.default:
            self._values.pop(key, None)
        else:
            self._values[key] = v
        if key.startswith("rule."):
            self._engine = None
        return v

    def reset(self) -> None:
        """Drop every override: back to the base rules."""
        self._values.clear()
        self._engine = None

    def engine(self) -> RuleEngine:
        """The rules engine to judge reps with: the base engine itself when no rule threshold
        is overridden, else one built from the base document with the overridden thresholds
        (the same object until a rule value changes)."""
        if not any(k.startswith("rule.") for k in self._values):
            return self.base
        if self._engine is None:
            doc = copy.deepcopy(self.base.doc)
            self._apply_rules(doc)
            self._engine = RuleEngine(doc)
        return self._engine

    def document(self) -> dict:
        """A deep copy of the base rules document with every override applied: rule thresholds,
        ``gate`` (``energy_threshold_ms2sq``, ``on_windows``, ``off_windows``) and
        ``rep_segmentation.min_range_deg``."""
        doc = copy.deepcopy(self.base.doc)
        self._apply_rules(doc)
        for key, s in SETTINGS.items():
            if key in self._values:
                v = self._values[key]
                section = doc.get(s.section)
                if not isinstance(section, dict):
                    section = doc[s.section] = {}
                section[s.field] = int(v) if key in WHOLE_NUMBER_KEYS else v
        return doc

    def save(self, path: Path) -> Path:
        """Write :meth:`document` to ``path`` as a rules YAML (loadable with
        :meth:`RuleEngine.from_yaml`), after a ``#`` header naming the base rules file and each
        changed key with its default and new value. Creates missing parent directories.
        Returns ``path``.

        Raises:
            ValueError: ``path`` is inside the packaged rules directory
                (``src/formcoach/rules/``); packaged rules are never overwritten.
        """
        path = Path(path)
        target = path.resolve()
        package = PACKAGE_RULES_DIR.resolve()
        if target == package or package in target.parents:
            raise ValueError(
                f"refusing to write {path}: packaged rules in {PACKAGE_RULES_DIR} are never "
                "overwritten; choose a path outside the package"
            )
        header = [f"# FormCoach rules: {self.base_name} with Bench tuning overrides."]
        if self._values:
            header.append("# Changed (default -> value):")
            for key, v in self._values.items():
                header.append(f"#   {key}: {self.default(key):g} -> {v:g}")
        else:
            header.append("# No values changed.")
        header.append("# Review before committing; nothing loads this file unless asked to.")
        body = yaml.safe_dump(self.document(), sort_keys=False, allow_unicode=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(header) + "\n" + body, encoding="utf-8")
        return path

    def _param(self, key: str) -> Param:
        try:
            return self._params[key]
        except KeyError:
            raise KeyError(f"unknown parameter {key!r}") from None

    def _apply_rules(self, doc: dict) -> None:
        """Write the overridden rule thresholds into ``doc`` (a copy of the base document)."""
        for ex, spec in doc.get("exercises", {}).items():
            for rule in spec.get("rules", []):
                for cond in rule["when"]:
                    key = rule_key(ex, rule["code"], cond["metric"])
                    if key in self._values:
                        cond["threshold"] = self._values[key]


def rejudge(reps: Sequence, engine: RuleEngine) -> list[list[dict]]:
    """The faults of each rep (a :class:`~formcoach.app.pipeline.RepMetrics`) under ``engine``,
    from its stored ``metrics`` and the requires-tags in its ``context``, as the fault dicts the
    pipeline stores. Changes nothing: the caller assigns them."""
    return [[f.as_dict() for f in engine.evaluate(r, context=set(r.context))] for r in reps]
