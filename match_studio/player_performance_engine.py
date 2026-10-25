"""Position-aware, transparent descriptive evaluation of featured-match players.

Uses only observed full-match synthetic player stats, not latent player ability.
Rankings are heuristic comparisons within position, not calibrated ratings.
"""
import numpy as np
import pandas as pd

# Stat weights are intentionally position-specific; only observed fields used.
WEIGHTS = {
    "GK": {"passes_completed": .3, "recoveries": .4},
    "CB": {"interceptions": 2, "tackles": 1.5, "clearances": 1.5, "recoveries": 1, "progressive_passes": .7},
    "WB": {"tackles": 1.5, "interceptions": 1.3, "recoveries": 1, "progressive_passes": 1.1, "key_passes": 1.2},
    "CM": {"progressive_passes": 1.5, "passes_completed": .2, "recoveries": 1, "interceptions": 1.1, "key_passes": 1.2},
    "AMW": {"goals": 4, "assists": 3, "key_passes": 1.5, "shots_on_target": 1.1, "successful_dribbles": 1.1, "progressive_passes": .5},
    "ST": {"goals": 4, "assists": 2.5, "shots_on_target": 1.3, "xg": 1.2, "key_passes": .7},
}


def evaluate_player_match(player_stats, players, min_minutes=45, *, fixture=None, lineups=None):
    if player_stats.empty:
        return {"available": False, "reason": "No individual match statistics"}
    if fixture is not None:
        teams = {str(fixture['home_team_id']), str(fixture['away_team_id'])}
        match_id = str(fixture['match_id'])
        if not player_stats['match_id'].astype(str).eq(match_id).all():
            raise ValueError("Featured player stats are stale: match ID differs from fixture")
        if not player_stats['team_id'].astype(str).isin(teams).all():
            raise ValueError("Featured player stats contain players from other teams")
    if lineups is not None:
        allowed = set(zip(lineups['player_id'].astype(str), lineups['team_id'].astype(str)))
        actual = set(zip(player_stats['player_id'].astype(str), player_stats['team_id'].astype(str)))
        if not actual.issubset(allowed):
            raise ValueError(f"Featured player stats contain nonparticipants: {actual - allowed}")
    df = player_stats.merge(players[["player_id", "player_name", "primary_position"]], on="player_id", how="left", validate="many_to_one")
    df = df[df.minutes_played >= min_minutes].copy()
    if df.empty:
        return {"available": False, "reason": "Insufficient player minutes"}
    if df.player_name.isna().any():
        return {"available": False, "reason": "Player names unavailable"}
    # Per-90, robustly normalized within each position in this fixture. Single GK
    # per side and limited observations make GK comparisons especially uncertain.
    scores = []
    for _, row in df.iterrows():
        pos = str(row.position_played) if "position_played" in df else str(row.primary_position)
        weights = WEIGHTS.get(pos, WEIGHTS["CM"])
        score = 0.0
        for field, weight in weights.items():
            if field in df:
                score += weight * float(row[field]) * 90.0 / max(float(row.minutes_played), 1)
        if "dispossessions" in df:
            score -= 0.7 * float(row.dispossessions) * 90 / max(float(row.minutes_played), 1)
        if "red_cards" in df:
            score -= 5 * float(row.red_cards)
        scores.append(score)
    df["raw_score"] = scores
    # Position-normalized within match: avoid rewarding striker scoring scales over CBs.
    df["position_score"] = df.groupby("position_played").raw_score.transform(
        lambda s: (s-s.mean()) / (s.std(ddof=0) if s.std(ddof=0) > 0 else 1.0))
    # More playing time increases confidence, without being a direct performance bonus.
    df["ranking_score"] = df.position_score * np.sqrt(df.minutes_played / 90.0)
    top = df.sort_values(["ranking_score", "minutes_played"], ascending=[False, False]).iloc[0]
    bottom = df.sort_values(["ranking_score", "minutes_played"], ascending=[True, False]).iloc[0]
    fields = ("goals", "assists", "shots", "shots_on_target", "xg", "key_passes", "passes_attempted",
              "passes_completed", "progressive_passes", "tackles", "interceptions", "recoveries",
              "clearances", "successful_dribbles", "dispossessions", "red_cards")
    def describe(row):
        pos = str(row.position_played)
        relevant = set(WEIGHTS.get(pos, {})) | {"goals", "assists", "minutes_played"}
        relevant |= ({"shots", "xg"} if pos in {"AMW", "ST"} else set())
        return {"player_name": str(row.player_name), "team_id": str(row.team_id),
                "position": pos, "minutes_played": int(row.minutes_played),
                "stats": {k: (round(float(row[k]), 2) if k == "xg" else int(row[k]))
                          for k in fields if k in relevant and k in row and pd.notna(row[k])}}
    return {"available": True, "player_of_the_match": describe(top),
            "most_disappointing_performer": describe(bottom),
            "method_note": "Heuristic position-relative ranking of synthetic observed full-match statistics; not a validated player rating. Goalkeeper shot-stopping metrics unavailable.",
            "eligible_player_count": len(df)}
