"""Start the paper-trading bot. Works with VS Code's Run button, a double-click or `python run_bot.py`.

If started with a Python that is not the project's .venv, it re-launches itself
with .venv so the installed packages (ccxt, python-telegram-bot) are found.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV_PY = ROOT / ".venv" / "Scripts" / "python.exe"


def main() -> int:
    if VENV_PY.exists() and Path(sys.executable).resolve() != VENV_PY.resolve():
        return subprocess.call([str(VENV_PY), "-m", "bot", "run", *sys.argv[1:]], cwd=ROOT)
    sys.path.insert(0, str(ROOT))
    from bot.__main__ import main as bot_main

    return bot_main(["run", *sys.argv[1:]])


if __name__ == "__main__":
    sys.exit(main())
