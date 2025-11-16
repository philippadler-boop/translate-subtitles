import io
import sys

from src.utils.progress import Progress


def test_progress_unknown_total_writes_output(monkeypatch):
    buf = io.StringIO()
    monkeypatch.setattr(sys, "stdout", buf)

    p = Progress("Task")
    p.start()  # total=None
    p.update(message="step 1")
    p.finish("done")

    out = buf.getvalue()
    assert "Task:" in out
    assert "step 1" in out
    assert "done" in out


def test_progress_known_total_reaches_100(monkeypatch):
    buf = io.StringIO()
    monkeypatch.setattr(sys, "stdout", buf)

    p = Progress("Task")
    p.start(total=2)
    p.update(message="first")
    p.update(message="second")
    p.finish("done")

    out = buf.getvalue()
    assert "[100%]" in out
