"""Step 4: derive historical team-half statistics from shots and team styles.

Inputs: data/historical_matches.parquet
        data/historical_team_shot_stats.parquet
        data/team_profiles.parquet
Output: data/historical_team_stats.parquet

Requires: numpy, pandas, pyarrow
Run: python generate_historical_team_stats.py
"""
from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"
SEED = 45

# A full match typically contains ~52-58 minutes of ball-in-play time.
MEAN_IN_PLAY_SECONDS_PER_HALF = 1650
SD_IN_PLAY_SECONDS_PER_HALF = 110
POSSESSION_STYLE_EFFECT = 0.014
POSSESSION_HOME_ADVANTAGE = 0.07
POSSESSION_NOISE_SD = 0.20
PASS_RATE_PER_POSSESSION_MINUTE = 13.0
PASS_COMPLETION_BASE = 0.79
PROGRESSIVE_PASS_SHARE_BASE = 0.085
MIN_POSSESSION_PCT = 20.0
MAX_POSSESSION_PCT = 80.0


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def generate_historical_team_stats(
    matches: pd.DataFrame,
    shot_stats: pd.DataFrame,
    profiles: pd.DataFrame,
    seed: int = SEED,
) -> pd.DataFrame:
    """One row per match/team/half. Shot data are authoritative."""
    rng = np.random.default_rng(seed)
    profile = profiles.set_index("team_id")
    required = {"match_id", "team_id", "period", "shots", "shots_on_target", "xg", "goals"}
    if not required.issubset(shot_stats.columns):
        raise ValueError(f"Missing shot columns: {required - set(shot_stats.columns)}")
    if shot_stats.duplicated(["match_id", "team_id", "period"]).any():
        raise ValueError("Duplicate team-half shot records")
    shots_index = shot_stats.set_index(["match_id", "team_id", "period"])
    rows = []

    for m in matches.itertuples(index=False):
        home, away = m.home_team_id, m.away_team_id
        if home not in profile.index or away not in profile.index:
            raise ValueError(f"Missing team profile for {m.match_id}")
        hp, ap = profile.loc[home], profile.loc[away]

        for half in (1, 2):
            # Generate complementary possession durations jointly for the two teams.
            in_play = int(np.clip(
                rng.normal(MEAN_IN_PLAY_SECONDS_PER_HALF, SD_IN_PLAY_SECONDS_PER_HALF),
                1250, 1950,
            ))
            style_diff = float(hp.possession_style - ap.possession_style)
            # Strength influences execution, but not as strongly as tactical style.
            quality_diff = float(hp.overall_strength - ap.overall_strength)
            logit = (POSSESSION_STYLE_EFFECT * style_diff
                     + 0.004 * quality_diff
                     + POSSESSION_HOME_ADVANTAGE
                     + rng.normal(0, POSSESSION_NOISE_SD))
            home_share = float(np.clip(sigmoid(logit),
                                       MIN_POSSESSION_PCT / 100,
                                       MAX_POSSESSION_PCT / 100))
            home_seconds = int(round(in_play * home_share))
            durations = {home: home_seconds, away: in_play - home_seconds}

            # Game state at start of half: optional, small effect on activity.
            home_lead_at_ht = int(m.ht_home_goals - m.ht_away_goals)
            for team, opp, is_home in [(home, away, True), (away, home, False)]:
                p, o = profile.loc[team], profile.loc[opp]
                key = (m.match_id, team, half)
                if key not in shots_index.index:
                    raise ValueError(f"Missing shot stats for {key}")
                s = shots_index.loc[key]
                possession_seconds = durations[team]
                possession_minutes = possession_seconds / 60.0
                possession_pct = 100.0 * possession_seconds / in_play

                tempo_factor = np.clip(1 + 0.004 * (float(p.tempo) - 55), 0.72, 1.35)
                directness_factor = np.clip(1 - 0.003 * (float(p.directness) - 55), 0.72, 1.25)
                # Trailing teams tend to play faster in H2; small average effect.
                trailing = (half == 2 and
                            ((is_home and home_lead_at_ht < 0) or
                             (not is_home and home_lead_at_ht > 0)))
                trailing_factor = 1.06 if trailing else 1.0
                expected_passes = (possession_minutes * PASS_RATE_PER_POSSESSION_MINUTE
                                   * tempo_factor * directness_factor * trailing_factor)
                passes_attempted = int(rng.poisson(max(20, expected_passes)))

                completion_rate = np.clip(
                    PASS_COMPLETION_BASE
                    + 0.0012 * (float(p.possession_style) - 55)
                    + 0.0007 * (float(p.overall_strength) - 65)
                    - 0.0008 * (float(o.pressing_intensity) - 55)
                    - 0.0008 * (float(p.directness) - 55)
                    + rng.normal(0, 0.018),
                    0.57, 0.94,
                )
                passes_completed = int(rng.binomial(passes_attempted, completion_rate))
                prog_share = np.clip(
                    PROGRESSIVE_PASS_SHARE_BASE
                    + 0.00045 * (float(p.directness) - 55)
                    + rng.normal(0, 0.009), 0.035, 0.17
                )
                progressive_passes = int(rng.binomial(passes_completed, prog_share))
                final_third_passes = int(rng.binomial(
                    progressive_passes, np.clip(rng.normal(0.67, 0.06), 0.40, 0.90)
                ))
                # Box entries reflect attacking volume and progression.
                box_entries = int(rng.poisson(max(
                    0.5, 0.60 * int(s.shots) + 0.13 * progressive_passes
                )))
                corners = int(rng.poisson(max(0.2, 0.22 * int(s.shots) + 0.5)))
                pressing = float(p.pressing_intensity)
                opp_poss_min = durations[opp] / 60.0
                tackles = int(rng.poisson(max(
                    0.5, opp_poss_min * (0.44 + 0.002 * (pressing - 55))
                )))
                interceptions = int(rng.poisson(max(
                    0.3, opp_poss_min * (0.28 + 0.001 * (pressing - 55))
                )))
                recoveries = int(rng.poisson(max(
                    1.0, opp_poss_min * (0.95 + 0.003 * (pressing - 55))
                )))
                clearances = int(rng.poisson(max(
                    0.4, 0.12 * float(shot_stats.loc[
                        (shot_stats.match_id == m.match_id) &
                        (shot_stats.team_id == opp) &
                        (shot_stats.period == half), "shots"
                    ].iloc[0]) + 1.8
                )))
                fouls = int(rng.poisson(max(1.0, 5.3 + 0.015 * (pressing - 55))))
                yellow_cards = int(rng.binomial(fouls, 0.09))
                red_cards = int(rng.binomial(yellow_cards, 0.018))

                rows.append({
                    "match_id": m.match_id,
                    "team_id": team,
                    "opponent_id": opp,
                    "period": half,
                    "is_home": is_home,
                    "possession_seconds": possession_seconds,
                    "in_play_seconds": in_play,
                    "possession_pct": possession_pct,
                    "shots": int(s.shots),
                    "shots_on_target": int(s.shots_on_target),
                    "xg": float(s.xg),
                    "goals": int(s.goals),
                    "passes_attempted": passes_attempted,
                    "passes_completed": passes_completed,
                    "progressive_passes": progressive_passes,
                    "passes_final_third": final_third_passes,
                    "box_entries": box_entries,
                    "corners": corners,
                    "tackles": tackles,
                    "interceptions": interceptions,
                    "recoveries": recoveries,
                    "clearances": clearances,
                    "fouls": fouls,
                    "yellow_cards": yellow_cards,
                    "red_cards": red_cards,
                })
    result = pd.DataFrame(rows)
    validate(result, matches, shot_stats)
    return result


def validate(stats: pd.DataFrame, matches: pd.DataFrame, shots: pd.DataFrame):
    assert len(stats) == 4 * len(matches)
    assert not stats.duplicated(["match_id", "team_id", "period"]).any()
    for (_, half), pair in stats.groupby(["match_id", "period"]):
        assert len(pair) == 2
        assert abs(pair.possession_pct.sum() - 100.0) < 1e-9
        assert pair.in_play_seconds.nunique() == 1
        assert pair.possession_seconds.sum() == pair.in_play_seconds.iloc[0]
    assert (stats.passes_completed <= stats.passes_attempted).all()
    assert (stats.progressive_passes <= stats.passes_completed).all()
    assert (stats.passes_final_third <= stats.progressive_passes).all()
    assert (stats.shots_on_target <= stats.shots).all()
    assert (stats.goals <= stats.shots_on_target).all()
    for col in ["box_entries", "corners", "tackles", "interceptions",
                "recoveries", "clearances", "fouls", "yellow_cards", "red_cards"]:
        assert (stats[col] >= 0).all(), col
    a = stats.set_index(["match_id", "team_id", "period"]).sort_index()
    b = shots.set_index(["match_id", "team_id", "period"]).sort_index()
    for col in ("shots", "shots_on_target", "goals"):
        assert a[col].equals(b[col]), col
    assert np.allclose(a.xg, b.xg)


def main():
    matches = pd.read_parquet(DATA_DIR / "historical_matches.parquet")
    shots = pd.read_parquet(DATA_DIR / "historical_team_shot_stats.parquet")
    profiles = pd.read_parquet(DATA_DIR / "team_profiles.parquet")
    result = generate_historical_team_stats(matches, shots, profiles)
    output = DATA_DIR / "historical_team_stats.parquet"
    result.to_parquet(output, index=False)
    print(f"Saved {len(result):,} team-half rows to {output}")
    print("\nAverage full-match totals per team:")
    numeric = ["shots", "xg", "passes_attempted", "passes_completed",
               "progressive_passes", "tackles", "interceptions", "corners"]
    print((result[numeric].sum() / (2 * len(matches))).round(2).to_string())
    print("\nSample:\n", result.head(4).to_string(index=False))


if __name__ == "__main__":
    main()
