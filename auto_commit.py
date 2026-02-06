"""
Auto-commit script — делает 3 коммита в день для зелёных клеточек на GitHub.
Запускается через Windows Task Scheduler каждый день.
"""
import subprocess
import os
import datetime
import random

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(REPO_DIR, "activity.log")

# Фразы для коммитов
MESSAGES = [
    "daily: update training logs",
    "daily: tune hyperparameters",
    "daily: review backtest results",
    "daily: optimize reward shaping",
    "daily: adjust indicators",
    "daily: analyze market data",
    "daily: improve agent performance",
    "daily: refactor trading env",
    "daily: update documentation",
    "daily: experiment with features",
    "daily: monitor agent metrics",
    "daily: data pipeline maintenance",
]


def run(cmd):
    """Run a shell command in the repo directory."""
    result = subprocess.run(
        cmd, cwd=REPO_DIR, shell=True,
        capture_output=True, text=True
    )
    return result.returncode, result.stdout.strip(), result.stderr.strip()


def do_commit(n):
    """Make one commit by appending to activity.log."""
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    msg = random.choice(MESSAGES)

    # Append line to log file
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{now}] commit #{n} — {msg}\n")

    run("git add -A")
    code, out, err = run(f'git commit -m "{msg}"')
    print(f"  Commit #{n}: {msg} -> {'OK' if code == 0 else err}")
    return code == 0


def main():
    print(f"=== Auto-commit — {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')} ===")

    # Make 3 commits
    success = 0
    for i in range(1, 4):
        if do_commit(i):
            success += 1

    # Push to GitHub
    code, out, err = run("git push")
    if code == 0:
        print(f"  Push OK ({success} commits)")
    else:
        print(f"  Push FAILED: {err}")

    print("=== Done ===\n")


if __name__ == "__main__":
    main()
