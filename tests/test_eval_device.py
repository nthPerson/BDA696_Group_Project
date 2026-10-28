"""`formcoach eval device`: on-device metrics parsed from a captured serial log."""

from __future__ import annotations

from formcoach.eval import device, report

LOG = """# FormCoach fw 1.1.1  chip=ESP32-S3 rev=0
# flash app=643792
# infer us=13400 arena=6196 class=0 p_active=0.02
# infer us=13450 arena=6196 class=2 p_active=0.91
"""


def test_parse_and_report_use_a_repo_relative_log_path(tmp_path):
    d = device.parse(LOG)
    assert d["n"] == 2 and d["firmware"] == "1.1.1" and d["flash"] == 643792

    log = report.REPO_ROOT / "reports" / "logs" / "_pytest_eval_device.log"
    log.write_text(LOG, encoding="utf-8")
    try:
        text = device.evaluate(log, out=tmp_path / "device.md").read_text(encoding="utf-8")
    finally:
        log.unlink()
    assert "--log reports/logs/_pytest_eval_device.log`" in text
    assert str(report.REPO_ROOT) not in text
    assert "13.43 / 13.45 / 13.45" in text


def test_repo_relative_keeps_paths_outside_the_repo(tmp_path):
    assert report.repo_relative(tmp_path / "x.log") == (tmp_path / "x.log").as_posix()
