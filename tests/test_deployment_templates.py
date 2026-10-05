from configparser import ConfigParser
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read_unit(name: str) -> ConfigParser:
    parser = ConfigParser(interpolation=None, strict=True)
    with (ROOT / "deploy" / name).open(encoding="utf-8") as source:
        parser.read_file(source)
    return parser


def test_systemd_backup_timer_targets_long_running_backup_service():
    service = _read_unit("rag-backup.service")
    timer = _read_unit("rag-backup.timer")

    assert service["Service"]["Type"] == "oneshot"
    assert service["Service"]["TimeoutStartSec"] == "0"
    assert timer["Timer"]["Unit"] == "rag-backup.service"
    assert timer["Timer"]["Persistent"] == "true"
    assert timer["Timer"]["OnCalendar"] == "*-*-* 02:00:00"
