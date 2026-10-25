"""Run every historical data generator in dependency order.

Place this file in data_generator/ beside the six generate_*.py scripts.
Run from anywhere: python data_generator/generate_data.py
"""
from pathlib import Path
import subprocess
import sys
import time

GENERATOR_DIR = Path(__file__).resolve().parent
STEPS = [
    "generate_teams.py",
    "generate_players.py",
    "generate_historical_matches.py",
    "generate_historical_team_stats.py",
    "generate_historical_player_match_stats.py",
    "generate_historical_player_season_stats.py",
]


def generate_data() -> None:
    missing = [name for name in STEPS if not (GENERATOR_DIR / name).is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing generator scripts in " + str(GENERATOR_DIR) + ": " + ", ".join(missing)
        )

    started = time.perf_counter()
    for index, script in enumerate(STEPS, start=1):
        print(f"\n{'=' * 65}", flush=True)
        print(f"[{index}/{len(STEPS)}] Running {script}", flush=True)
        print(f"{'=' * 65}", flush=True)
        subprocess.run(
            [sys.executable, str(GENERATOR_DIR / script)],
            cwd=GENERATOR_DIR,
            check=True,
        )

    print(f"\nAll {len(STEPS)} generators finished successfully "
          f"in {time.perf_counter() - started:.1f} seconds.", flush=True)
    print(f"Data directory: {GENERATOR_DIR / 'data'}", flush=True)


if __name__ == "__main__":
    try:
        generate_data()
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        print(f"\nGeneration stopped: {exc}", file=sys.stderr)
        sys.exit(1)
