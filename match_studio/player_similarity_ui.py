"""Phase-safe, position-specific player similarity for the Match Studio UI.

Uses only historical_player_match_stats already filtered before the featured
fixture by player_performance.load_player_history. Does not use future matches.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

POSITION_FEATURES = {
    "CB": ["tackles", "interceptions", "recoveries", "clearances", "progressive_passes", "passes_attempted"],
    "WB": ["tackles", "interceptions", "recoveries", "progressive_passes", "progressive_carries", "successful_dribbles"],
    "CM": ["progressive_passes", "key_passes", "recoveries", "interceptions", "progressive_carries", "passes_attempted"],
    "AMW": ["xg", "shots", "assists", "xa", "key_passes", "progressive_carries", "successful_dribbles"],
    "CF": ["xg", "shots", "goals", "assists", "key_passes", "shots_on_target"],
    "ST": ["xg", "shots", "goals", "assists", "key_passes", "shots_on_target"],
}
LABELS = {
    "tackles": "Tackles /90", "interceptions": "Interceptions /90", "recoveries": "Recoveries /90",
    "clearances": "Clearances /90", "progressive_passes": "Progressive passes /90",
    "passes_attempted": "Passes attempted /90", "progressive_carries": "Progressive carries /90",
    "successful_dribbles": "Successful dribbles /90", "key_passes": "Key passes /90",
    "xg": "xG /90", "shots": "Shots /90", "goals": "Goals /90",
    "assists": "Assists /90", "xa": "xA /90", "shots_on_target": "Shots on target /90",
}


def similar_players(history, reference_id, reference_team_id, positions, names, top_n=5, last_n=10):
    """Return same-position players from other clubs with PCA cosine similarity.

    The displayed percentage maps cosine similarity [-1,1] to [0,100].
    The comparison is relative to the position pool, not an absolute rating.
    """
    reference_id = str(reference_id)
    position = positions.get(reference_id)
    if position not in POSITION_FEATURES:
        raise ValueError("This position is not supported by the available metrics")
    required = {"player_id", "team_id", "minutes_played", "match_id"}
    if not required.issubset(history.columns):
        raise ValueError(f"Missing historical fields: {sorted(required - set(history.columns))}")
    fields = [f for f in POSITION_FEATURES[position] if f in history.columns]
    if len(fields) < 3:
        raise ValueError("Insufficient position-specific statistics")
    rows = history.copy()
    rows["player_id"] = rows.player_id.astype(str)
    rows["team_id"] = rows.team_id.astype(str)
    rows = rows[rows.player_id.isin({pid for pid, pos in positions.items() if pos == position})]
    if "_match_date" in rows:
        rows = rows.sort_values(["_match_date", "match_id"])
    rows = rows.groupby("player_id", group_keys=False).tail(last_n)
    for f in fields + ["minutes_played"]:
        rows[f] = pd.to_numeric(rows[f], errors="coerce").fillna(0)
    grouped = rows.groupby("player_id", sort=False)
    profiles = grouped.agg(team_id=("team_id", "last"), Matches=("match_id", "nunique"),
                           Minutes=("minutes_played", "sum"))
    for f in fields:
        profiles[f] = grouped[f].sum() * 90 / profiles.Minutes.replace(0, np.nan)
    profiles = profiles[(profiles.Minutes >= 180) | (profiles.index == reference_id)].copy()
    if reference_id not in profiles.index or profiles.loc[reference_id, "Minutes"] <= 0:
        raise ValueError("The selected player has no historical minutes")
    if len(profiles) < 3:
        raise ValueError("Not enough players with sufficient historical appearances")
    features = profiles[fields].replace([np.inf, -np.inf], np.nan).fillna(0)
    scaled = StandardScaler().fit_transform(features)
    max_components = min(scaled.shape[0] - 1, scaled.shape[1])
    if max_components < 1 or np.allclose(scaled, 0):
        raise ValueError("Not enough statistical variation for comparison")
    coords = PCA(n_components=0.90, svd_solver="full").fit_transform(scaled)
    idx = profiles.index.get_loc(reference_id)
    similarity = cosine_similarity(coords[idx:idx+1], coords).ravel()
    profiles["Similarity (%)"] = np.round((similarity + 1) * 50, 1)
    profiles = profiles[(profiles.index != reference_id) & (profiles.team_id != str(reference_team_id))]
    profiles = profiles.sort_values("Similarity (%)", ascending=False).head(int(top_n)).copy()
    profiles["player_id"] = profiles.index.astype(str)
    profiles["Player"] = [names.get(pid, pid) for pid in profiles.index]
    profiles["Position"] = position
    for f in fields:
        profiles[LABELS[f]] = profiles[f].round(2)
    return profiles.reset_index(drop=True)


def radar_profiles(history, reference_id, comparison_id, positions, last_n=10):
    """Historical per-90 radar values expressed as position-pool percentile ranks.

    Both players and the reference population use the same pre-fixture history
    and recent-appearance window as the similarity table.
    """
    reference_id, comparison_id = str(reference_id), str(comparison_id)
    position = positions.get(reference_id)
    if position not in POSITION_FEATURES or positions.get(comparison_id) != position:
        raise ValueError("Radar players must have the same supported position")
    fields = [f for f in POSITION_FEATURES[position] if f in history.columns]
    if len(fields) < 3:
        raise ValueError("Insufficient metrics for radar")
    rows = history.copy()
    rows["player_id"] = rows.player_id.astype(str)
    rows = rows[rows.player_id.isin({str(pid) for pid, pos in positions.items() if pos == position})]
    if "_match_date" in rows:
        rows = rows.sort_values(["_match_date", "match_id"])
    rows = rows.groupby("player_id", group_keys=False).tail(last_n).copy()
    for field in fields + ["minutes_played"]:
        rows[field] = pd.to_numeric(rows[field], errors="coerce").fillna(0)
    grouped = rows.groupby("player_id")
    minutes = grouped.minutes_played.sum()
    pool = pd.DataFrame(index=minutes.index)
    for field in fields:
        pool[field] = grouped[field].sum() * 90 / minutes.replace(0, np.nan)
    pool = pool.loc[minutes >= 180].replace([np.inf, -np.inf], np.nan).fillna(0)
    if reference_id not in pool.index or comparison_id not in pool.index:
        raise ValueError("Both players need at least 180 historical minutes for a comparable radar")
    # Percentile is a descriptive position-pool rank, NOT an ability rating.
    ranks = pool.rank(axis=0, pct=True) * 100
    return pd.DataFrame({
        "Metric": [LABELS[f] for f in fields],
        "Reference percentile": [round(float(ranks.at[reference_id, f]), 1) for f in fields],
        "Comparison percentile": [round(float(ranks.at[comparison_id, f]), 1) for f in fields],
        "Reference /90": [round(float(pool.at[reference_id, f]), 2) for f in fields],
        "Comparison /90": [round(float(pool.at[comparison_id, f]), 2) for f in fields],
    })
