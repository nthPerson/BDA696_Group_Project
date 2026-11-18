"""Re-export of :mod:`formcoach.pose.synthetic` (the synthetic figure moved into the package so the
Bench can use it); kept so the tests' ``from .synthetic_pose import ...`` lines keep working."""

from formcoach.pose.synthetic import (  # noqa: F401
    FOREARM,
    LOWER_LEG,
    UPPER_ARM,
    UPPER_LEG,
    curl_deg,
    pose_frame,
)
