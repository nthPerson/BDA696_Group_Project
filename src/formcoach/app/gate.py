"""Camera gate implementations (docs/02 §4.2): ``always_on``, ``energy``, ``laptop``, ``device``.

All gates consume :class:`SISample` one at a time via ``update`` and return the current gate
state. Window-based gates decide every ``stride`` samples over the last ``window`` samples and
pass the decision through :class:`Hysteresis` (open after ``GATE_ON_WINDOWS`` consecutive
active windows, close after ``GATE_OFF_WINDOWS`` idle ones — the same constants as the firmware).
"""

from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np

from formcoach.io.protocol import (
    GATE_ENERGY_THRESHOLD_MS2SQ,
    GATE_OFF_WINDOWS,
    GATE_ON_WINDOWS,
    WINDOW_SAMPLES,
    WINDOW_STRIDE_SAMPLES,
)
from formcoach.io.source import SISample

# var(|a|) in (m/s²)², balanced-accuracy fit on RecoFit windows; one constant shared with the
# firmware through protocol.py / protocol.h (ADR-0019)
DEFAULT_ENERGY_THRESHOLD = GATE_ENERGY_THRESHOLD_MS2SQ
DEFAULT_LAPTOP_MODEL = Path(__file__).resolve().parents[3] / "models" / "gate_rf.joblib"
GATE_NAMES = ("always_on", "energy", "laptop", "device")


class Hysteresis:
    def __init__(self, on_windows: int = GATE_ON_WINDOWS, off_windows: int = GATE_OFF_WINDOWS):
        self.on_windows, self.off_windows = on_windows, off_windows
        self.state = False
        self._run = 0

    def update(self, active: bool) -> bool:
        if active == self.state:
            self._run = 0
            return self.state
        self._run += 1
        need = self.on_windows if active else self.off_windows
        if self._run >= need:
            self.state = active
            self._run = 0
        return self.state


class GateBase:
    name = "base"

    def __init__(self) -> None:
        self.state = False

    def update(self, sample: SISample) -> bool:  # pragma: no cover - interface
        raise NotImplementedError


class AlwaysOnGate(GateBase):
    name = "always_on"

    def update(self, sample: SISample) -> bool:
        self.state = True
        return True


class DeviceGate(GateBase):
    """Trust ``flags.gate_state`` from the wearable (hysteresis already applied on-device)."""

    name = "device"

    def update(self, sample: SISample) -> bool:
        self.state = sample.gate_state
        return self.state


class _WindowGate(GateBase):
    def __init__(
        self,
        window: int = WINDOW_SAMPLES,
        stride: int = WINDOW_STRIDE_SAMPLES,
        on_windows: int = GATE_ON_WINDOWS,
        off_windows: int = GATE_OFF_WINDOWS,
    ):
        super().__init__()
        self.buf: deque[tuple[float, float, float, float, float, float]] = deque(maxlen=window)
        self.window, self.stride = window, stride
        self.hyst = Hysteresis(on_windows, off_windows)
        self._n = 0

    def decide(self, x: np.ndarray) -> bool:  # pragma: no cover - interface
        raise NotImplementedError

    def update(self, sample: SISample) -> bool:
        self.buf.append((sample.ax, sample.ay, sample.az, sample.gx, sample.gy, sample.gz))
        self._n += 1
        if len(self.buf) == self.window and self._n % self.stride == 0:
            self.state = self.hyst.update(self.decide(np.asarray(self.buf, dtype=np.float32)))
        return self.state


class EnergyGate(_WindowGate):
    """``var(|a|) > threshold`` over the window (the simplest baseline)."""

    name = "energy"

    def __init__(self, threshold: float = DEFAULT_ENERGY_THRESHOLD, **kw):
        super().__init__(**kw)
        self.threshold = threshold

    def decide(self, x: np.ndarray) -> bool:
        amag = np.linalg.norm(x[:, :3], axis=1)
        return bool(amag.var() > self.threshold)


class LaptopGate(_WindowGate):
    """The RF window model (``train gate --model rf``) run on the laptop from the raw stream."""

    name = "laptop"

    def __init__(self, model_path: Path = DEFAULT_LAPTOP_MODEL, **kw):
        super().__init__(**kw)
        import joblib

        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"{model_path} not found: run `formcoach train gate --model rf` first, or use "
                "--gate energy"
            )
        self.model = joblib.load(model_path)
        from formcoach.signal.features import window_features

        self._features = window_features

    def decide(self, x: np.ndarray) -> bool:
        f = np.array([list(self._features(x, 50.0).values())], dtype=np.float32)
        return bool(int(self.model.predict(f)[0]) == 1)


def make_gate(name: str, **kw) -> GateBase:
    if name == "always_on":
        return AlwaysOnGate()
    if name == "energy":
        return EnergyGate(**kw)
    if name == "laptop":
        return LaptopGate(**kw)
    if name == "device":
        return DeviceGate()
    raise ValueError(f"unknown gate {name!r}; choose from {GATE_NAMES}")
