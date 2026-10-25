"""Step 3: generate historical fixtures, shots/xG, and scores derived from shots.

Inputs (from generate_teams.py): data/teams.parquet, data/team_profiles.parquet
Outputs: data/historical_matches.parquet, data/historical_shots.parquet,
         data/historical_team_shot_stats.parquet

Install: pip install numpy pandas pyarrow
Run: python generate_historical_matches.py
"""

from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"
SEED = 47
SEASON_START_YEARS = (2024, 2025)
START_MONTH_DAY = (8, 16)
ROUND_SPACING_DAYS = 7

# Adjustable simulation parameters. Expected goals are generated from shots,
# not sampled directly. These are starting calibration values.
BASE_SHOTS_PER_TEAM = 10.8
HOME_SHOT_MULTIPLIER = 1.08
STRENGTH_SHOT_SENSITIVITY = 0.012
STRENGTH_QUALITY_SENSITIVITY = 0.005
BASE_MEAN_XG_PER_SHOT = 0.105
SHOT_QUALITY_CONCENTRATION = 10.0
FIRST_HALF_SHOT_SHARE = 0.46
MATCH_TEMPO_SIGMA = 0.12
SEASON_STRENGTH_SHIFT_SD = 2.5
MIN_SHOT_RATE, MAX_SHOT_RATE = 3.0, 27.0
MIN_MEAN_XG, MAX_MEAN_XG = 0.045, 0.22


def round_robin(team_ids: list[str]) -> list[list[tuple[str, str]]]:
    """Create home-and-away fixtures using the circle method."""
    if len(team_ids) % 2 or len(set(team_ids)) != len(team_ids):
        raise ValueError("Expected an even number of unique teams")
    rotation = team_ids.copy()
    first_leg = []
    for rnd in range(len(team_ids) - 1):
        pairs = []
        for j in range(len(team_ids) // 2):
            a, b = rotation[j], rotation[-1 - j]
            pairs.append((a, b) if (rnd + j) % 2 == 0 else (b, a))
        first_leg.append(pairs)
        rotation = [rotation[0], rotation[-1], *rotation[1:-1]]
    return first_leg + [[(away, home) for home, away in rnd] for rnd in first_leg]


def simulate_shots(
    rng: np.random.Generator,
    match_id: str,
    team_id: str,
    opponent_id: str,
    half: int,
    expected_shots: float,
    mean_xg: float,
) -> list[dict]:
    """Simulate shots and Bernoulli goal outcomes for one team in one half.

    A goal is always on target. Non-goal shots may be on target or off target.
    xG describes pre-shot goal probability, not post-shot outcome.
    """
    count = int(rng.poisson(max(0.0, expected_shots)))
    concentration = SHOT_QUALITY_CONCENTRATION
    alpha = mean_xg * concentration
    beta = (1.0 - mean_xg) * concentration
    rows = []
    for _ in range(count):
        minute = int(rng.integers(1 if half == 1 else 46, 46 if half == 1 else 91))
        second = int(rng.integers(0, 60))
        xg = float(np.clip(rng.beta(alpha, beta), 0.005, 0.85))
        is_goal = bool(rng.random() < xg)
        # Simple on-target approximation; later can be calibrated by shot type.
        on_target = bool(is_goal or rng.random() < 0.34)
        rows.append(
            {
                "match_id": match_id,
                "team_id": team_id,
                "opponent_id": opponent_id,
                "period": half,
                "minute": minute,
                "second": second,
                "xg": xg,
                "is_on_target": on_target,
                "is_goal": is_goal,
            }
        )
    return rows


def generate_historical_matches(
    teams: pd.DataFrame, profiles: pd.DataFrame, seed: int = SEED
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    ids = teams["team_id"].tolist()
    profile = profiles.set_index("team_id")
    if len(ids) != len(set(ids)) or not set(ids).issubset(set(profile.index)):
        raise ValueError("Duplicate team IDs or missing team profiles")

    match_rows, shot_rows = [], []
    for season_start in SEASON_START_YEARS:
        shifts = dict(zip(ids, rng.normal(0, SEASON_STRENGTH_SHIFT_SD, len(ids))))
        schedule = round_robin(list(rng.permutation(ids)))
        season_id = f"{season_start}-{str(season_start + 1)[-2:]}"
        start = pd.Timestamp(season_start, *START_MONTH_DAY)
        for round_no, games in enumerate(schedule, start=1):
            for game_no, (home, away) in enumerate(games, start=1):
                match_id = f"H{season_start}{(round_no - 1) * len(games) + game_no:04d}"
                tempo_factor = float(rng.lognormal(-0.5 * MATCH_TEMPO_SIGMA**2, MATCH_TEMPO_SIGMA))
                match_shots = []
                for team, opponent, is_home in [(home, away, True), (away, home, False)]:
                    attack = float(profile.loc[team, "attack_strength"]) + shifts[team]
                    defense = float(profile.loc[opponent, "defense_strength"]) + shifts[opponent]
                    diff = attack - defense
                    # Team style affects chance volume but does not dictate quality.
                    team_tempo = float(profile.loc[team, "tempo"])
                    volume = BASE_SHOTS_PER_TEAM * np.exp(STRENGTH_SHOT_SENSITIVITY * diff)
                    volume *= 1.0 + 0.002 * (team_tempo - 50.0)
                    volume *= HOME_SHOT_MULTIPLIER if is_home else 1.0 / HOME_SHOT_MULTIPLIER
                    volume *= tempo_factor
                    volume = float(np.clip(volume, MIN_SHOT_RATE, MAX_SHOT_RATE))
                    # Direct teams can produce higher-quality chances without dominating possession.
                    directness = float(profile.loc[team, "directness"])
                    quality = BASE_MEAN_XG_PER_SHOT * np.exp(STRENGTH_QUALITY_SENSITIVITY * diff)
                    quality *= 1.0 + 0.001 * (directness - 50.0)
                    quality = float(np.clip(quality, MIN_MEAN_XG, MAX_MEAN_XG))
                    for half, fraction in [(1, FIRST_HALF_SHOT_SHARE), (2, 1 - FIRST_HALF_SHOT_SHARE)]:
                        match_shots.extend(
                            simulate_shots(rng, match_id, team, opponent, half, volume * fraction, quality)
                        )

                match_shots.sort(key=lambda s: (s["minute"], s["second"], s["team_id"]))
                for i, shot in enumerate(match_shots, start=1):
                    shot["shot_id"] = f"{match_id}_S{i:03d}"
                shot_rows.extend(match_shots)
                home_goals_h1 = sum(s["is_goal"] for s in match_shots if s["team_id"] == home and s["period"] == 1)
                away_goals_h1 = sum(s["is_goal"] for s in match_shots if s["team_id"] == away and s["period"] == 1)
                home_goals = sum(s["is_goal"] for s in match_shots if s["team_id"] == home)
                away_goals = sum(s["is_goal"] for s in match_shots if s["team_id"] == away)
                goals = [s for s in match_shots if s["is_goal"]]
                match_rows.append(
                    {
                        "match_id": match_id,
                        "season_id": season_id,
                        "competition_id": "LEAGUE_01",
                        "round": round_no,
                        "date": start + pd.Timedelta(days=ROUND_SPACING_DAYS * (round_no - 1)),
                        "home_team_id": home,
                        "away_team_id": away,
                        "ht_home_goals": home_goals_h1,
                        "ht_away_goals": away_goals_h1,
                        "ft_home_goals": home_goals,
                        "ft_away_goals": away_goals,
                        "first_goal_team_id": goals[0]["team_id"] if goals else None,
                        "first_goal_minute": goals[0]["minute"] if goals else None,
                    }
                )

    matches = pd.DataFrame(match_rows).sort_values(["date", "match_id"]).reset_index(drop=True)
    shots = pd.DataFrame(
        shot_rows,
        columns=[
            "shot_id",
            "match_id",
            "team_id",
            "opponent_id",
            "period",
            "minute",
            "second",
            "xg",
            "is_on_target",
            "is_goal",
        ],
    )
    if not shots.empty:
        shots = shots.sort_values(["match_id", "minute", "second", "shot_id"]).reset_index(drop=True)

    # Include zero-shot halves explicitly, to guarantee four rows per match.
    base = []
    for m in matches.itertuples(index=False):
        for team in (m.home_team_id, m.away_team_id):
            for half in (1, 2):
                base.append({"match_id": m.match_id, "team_id": team, "period": half})
    stats = pd.DataFrame(base)
    if not shots.empty:
        agg = shots.groupby(["match_id", "team_id", "period"], as_index=False).agg(
            shots=("shot_id", "count"),
            shots_on_target=("is_on_target", "sum"),
            xg=("xg", "sum"),
            goals=("is_goal", "sum"),
        )
        stats = stats.merge(agg, how="left", on=["match_id", "team_id", "period"])
    else:
        for field in ("shots", "shots_on_target", "xg", "goals"):
            stats[field] = 0
    for field in ("shots", "shots_on_target", "goals"):
        stats[field] = stats[field].fillna(0).astype(int)
    stats["xg"] = stats["xg"].fillna(0.0)
    validate(matches, shots, stats, ids)
    return matches, shots, stats


def validate(matches: pd.DataFrame, shots: pd.DataFrame, stats: pd.DataFrame, team_ids: list[str]) -> None:
    assert matches.match_id.is_unique
    assert len(stats) == 4 * len(matches)
    assert (matches.home_team_id != matches.away_team_id).all()
    assert (matches.ft_home_goals >= matches.ht_home_goals).all()
    assert (matches.ft_away_goals >= matches.ht_away_goals).all()
    assert (stats.goals <= stats.shots_on_target).all()
    assert (stats.shots_on_target <= stats.shots).all()
    assert (stats.xg >= 0).all()
    assert shots.shot_id.is_unique
    assert shots.xg.between(0, 1).all()
    assert (~shots.is_goal | shots.is_on_target).all()
    assert (
        ((shots.period == 1) & shots.minute.between(1, 45)) | ((shots.period == 2) & shots.minute.between(46, 90))
    ).all()
    n = len(team_ids)
    for season_id, season in matches.groupby("season_id"):
        assert len(season) == n * (n - 1), season_id
        assert season["round"].nunique() == 2 * (n - 1)
        for _, rnd in season.groupby("round"):
            assert len(rnd) == n // 2
            assert set(rnd.home_team_id).isdisjoint(set(rnd.away_team_id))
            assert set(rnd.home_team_id) | set(rnd.away_team_id) == set(team_ids)
        assert season.groupby(["home_team_id", "away_team_id"]).size().eq(1).all()
    indexed = stats.set_index(["match_id", "team_id", "period"])
    for m in matches.itertuples(index=False):
        for team, half_score, full_score in [
            (m.home_team_id, m.ht_home_goals, m.ft_home_goals),
            (m.away_team_id, m.ht_away_goals, m.ft_away_goals),
        ]:
            assert int(indexed.loc[(m.match_id, team, 1), "goals"]) == half_score
            assert (
                int(indexed.loc[(m.match_id, team, 1), "goals"] + indexed.loc[(m.match_id, team, 2), "goals"])
                == full_score
            )


def main() -> None:
    teams = pd.read_parquet(DATA_DIR / "teams.parquet")
    profiles = pd.read_parquet(DATA_DIR / "team_profiles.parquet")
    matches, shots, stats = generate_historical_matches(teams, profiles)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name, df in [
        ("historical_matches", matches),
        ("historical_shots", shots),
        ("historical_team_shot_stats", stats),
    ]:
        path = DATA_DIR / f"{name}.parquet"
        df.to_parquet(path, index=False)
        print(f"Saved {len(df):,} rows: {path}")
    print(f"\nGoals per match: {(matches.ft_home_goals + matches.ft_away_goals).mean():.2f}")
    print(f"Shots per match: {len(shots) / len(matches):.2f}")
    print(f"xG per match: {stats.xg.sum() / len(matches):.2f}")
    print("\nSample fixtures:\n", matches.head().to_string(index=False))


if __name__ == "__main__":
    main()
