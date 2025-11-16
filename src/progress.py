import sys
import time
from typing import Optional


class Progress:
    """Simple terminal progress helper.

    Usage:
      p = Progress("Extracting audio")
      p.start(total=3)
      p.update(1, "extracting chunk 1")
      p.finish("done")

    This is intentionally lightweight and avoids external deps so it works on
    Windows PowerShell and plain terminals.
    """

    def __init__(self, title: str = "Progress") -> None:
        self.title = title
        self.total: Optional[int] = None
        self.current = 0
        self.last_print_len = 0
        self.start_time = None

    def start(self, total: Optional[int] = None) -> None:
        self.total = total
        self.current = 0
        self.start_time = time.time()
        self._print_line(0, "starting")

    def update(self, step: int = 1, message: str = "") -> None:
        if self.total is None:
            # unknown total: increment and display count
            self.current += step
            pct = None
        else:
            self.current = min(self.total, self.current + step)
            pct = int((self.current / self.total) * 100) if self.total else 0

        self._print_line(pct, message)

    def set_total(self, total: int) -> None:
        self.total = total

    def finish(self, message: str = "done") -> None:
        if self.total is not None:
            self.current = self.total
            pct = 100
        else:
            pct = None
        self._print_line(pct, message)
        sys.stdout.write("\n")
        sys.stdout.flush()

    def _print_line(self, pct: Optional[int], message: str) -> None:
        if pct is None:
            text = f"{self.title}: {self.current} - {message}"
        else:
            text = f"{self.title}: [{pct:3d}%] {self.current}/{self.total} - {message}"

        # erase previous line
        clear = "\r" + " " * max(self.last_print_len, len(text)) + "\r"
        sys.stdout.write(clear)
        sys.stdout.write(text)
        sys.stdout.flush()
        self.last_print_len = len(text)
