"""IMU sources and session recording.

``protocol`` mirrors ``firmware/include/protocol.h`` (packet layout, UUIDs, commands).
Planned (Checkpoint 4): ``ble.py`` (bleak), ``serial.py`` (pyserial), ``replay.py``
(Parquet/CSV replay, the no-hardware path) and ``recorder.py`` (``data/team/<S#>/<session>/``).
All sources implement ``iter_samples()`` yielding the same ``Sample`` dataclass.
"""
