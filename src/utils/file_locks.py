"""Cross-platform file locking utilities."""

import contextlib
import sys
from pathlib import Path

if sys.platform == "win32":
    import msvcrt

    @contextlib.contextmanager
    def file_lock(path: Path):
        """Windows file locking using msvcrt.

        Args:
            path: Path to the file being protected (lock file created alongside)

        Yields:
            None - use as context manager around file operations
        """
        lock_path = Path(str(path) + ".lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with open(lock_path, "w") as lock_file:
            try:
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
                yield
            finally:
                try:
                    msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
                except OSError:
                    pass

else:
    import fcntl

    @contextlib.contextmanager
    def file_lock(path: Path):
        """Unix file locking using fcntl.

        Args:
            path: Path to the file being protected (lock file created alongside)

        Yields:
            None - use as context manager around file operations
        """
        lock_path = Path(str(path) + ".lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with open(lock_path, "w") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
