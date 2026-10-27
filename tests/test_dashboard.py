"""The dashboard renders every tab without an exception."""
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app" / "dashboard.py")


def test_dashboard_runs():
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    assert not at.exception
    assert len(at.tabs) == 7
