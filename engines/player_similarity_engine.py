"""Player similarity from observed historical seasons (no hidden player ratings).

Run: python player_similarity_engine.py
Requires: pandas, numpy, pyarrow, scikit-learn
Input: data/historical_player_season_stats.parquet
Output: data/player_similarity_profiles.parquet, data/player_similarity_top10.parquet
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path(__file__).resolve().parent.parent / "data_generator" / "data"
RECENT_SEASON_WEIGHT = 0.65
MIN_SEASON_MINUTES = 450
MIN_TOTAL_MINUTES = 900
N_COMPONENTS = 0.90  # preserve 90% variance, up to number of features
TOP_N = 10

FEATURE_GROUPS = {
    "attacking": ["shots_p90", "xg_p90", "goals_p90", "assists_p90", "xa_p90", "key_passes_p90"],
    "passing": ["passes_attempted_p90", "progressive_passes_p90", "passes_final_third_p90", "pass_completion_pct"],
    "defending": ["tackles_p90", "interceptions_p90", "recoveries_p90", "clearances_p90"],
    "possession": ["progressive_carries_p90", "successful_dribbles_p90", "dispossessions_p90"],
    "physical": ["distance_km_p90", "sprints_p90", "accelerations_p90"],
}
DEFAULT_WEIGHTS = {name: 1.0 for name in FEATURE_GROUPS}


def build_profiles(season: pd.DataFrame) -> pd.DataFrame:
    if "season_id" not in season or "primary_position" not in season:
        raise ValueError("Season stats need season_id and primary_position")
    features = list(dict.fromkeys(f for group in FEATURE_GROUPS.values() for f in group))
    missing = set(features + ["player_id", "minutes_played"]) - set(season.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    seasons = sorted(season.season_id.dropna().unique())
    if len(seasons) < 2:
        raise ValueError("Expected two historical seasons")
    latest = seasons[-1]
    rows = []
    for player_id, g in season.groupby("player_id", sort=False):
        valid = g.loc[g.minutes_played >= MIN_SEASON_MINUTES].copy()
        if valid.empty or g.minutes_played.sum() < MIN_TOTAL_MINUTES:
            continue
        # Prefer recent observations, but weight each season by available minutes.
        valid["blend_weight"] = valid.minutes_played * np.where(
            valid.season_id == latest, RECENT_SEASON_WEIGHT, 1.0 - RECENT_SEASON_WEIGHT
        )
        weights = valid.blend_weight.to_numpy(dtype=float)
        row = {
            "player_id": player_id,
            "primary_position": g.primary_position.iloc[0],
            "total_minutes": int(g.minutes_played.sum()),
            "latest_season": str(latest),
        }
        for field in ["player_name", "name", "team_id", "age"]:
            if field in g:
                row[field] = g[field].iloc[-1]
        for feature in features:
            vals = pd.to_numeric(valid[feature], errors="coerce").to_numpy(dtype=float)
            ok = np.isfinite(vals)
            row[feature] = float(np.average(vals[ok], weights=weights[ok])) if ok.any() else np.nan
        rows.append(row)
    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError("No players pass the playing-time thresholds")
    return result


def rank_similar_players(
    profiles: pd.DataFrame, player_id, weights=None, top_n=TOP_N, position_only=True
) -> pd.DataFrame:
    """Fit PCA in a position group and return cosine similarity for a reference player."""
    weights = DEFAULT_WEIGHTS if weights is None else weights
    unknown = set(weights) - set(FEATURE_GROUPS)
    if unknown:
        raise ValueError(f"Unknown categories: {sorted(unknown)}")
    if any(float(v) < 0 for v in weights.values()) or sum(map(float, weights.values())) <= 0:
        raise ValueError("Weights must be nonnegative, with at least one positive weight")
    target = profiles.loc[profiles.player_id.astype(str) == str(player_id)]
    if len(target) != 1:
        raise ValueError(f"Expected one reference player, found {len(target)} for {player_id}")
    position = target.primary_position.iloc[0]
    if position == "GK":
        raise ValueError("Goalkeepers require dedicated save/shot-stopping features (not generated yet)")
    pool = profiles.loc[profiles.primary_position == position].copy() if position_only else profiles.copy()
    if len(pool) < 3:
        raise ValueError("Not enough comparable players")
    selected = [
        (feature, float(weights.get(category, 0)))
        for category, feats in FEATURE_GROUPS.items()
        for feature in feats
        if float(weights.get(category, 0)) > 0
    ]
    features = [f for f, _ in selected]
    # Median imputation is computed within comparison pool, not using hidden ratings.
    x = pool[features].replace([np.inf, -np.inf], np.nan)
    x = x.fillna(x.median()).fillna(0)
    scaled = StandardScaler().fit_transform(x)
    # Weight categories equally regardless of number of features in each group.
    per_feature_weights = np.array(
        [
            np.sqrt(weight / len(FEATURE_GROUPS[next(k for k, v in FEATURE_GROUPS.items() if f in v)]))
            for f, weight in selected
        ]
    )
    weighted = scaled * per_feature_weights
    n_max = min(weighted.shape[0] - 1, weighted.shape[1])
    if n_max < 1 or np.allclose(weighted, 0):
        raise ValueError("Insufficient variance for PCA")
    coords = PCA(
        n_components=min(N_COMPONENTS, n_max) if isinstance(N_COMPONENTS, int) else N_COMPONENTS, svd_solver="full"
    ).fit_transform(weighted)
    # Centered PCA coordinates can have zero norm for rare degenerate cases.
    ref_idx = np.flatnonzero(pool.player_id.astype(str).to_numpy() == str(player_id))[0]
    scores = cosine_similarity(coords[ref_idx : ref_idx + 1], coords).ravel()
    pool["similarity"] = scores
    pool = pool.loc[pool.player_id.astype(str) != str(player_id)]
    pool = pool.sort_values("similarity", ascending=False).head(top_n).copy()
    pool.insert(0, "reference_player_id", player_id)
    pool["comparison_position"] = position
    return pool[
        ["reference_player_id", "player_id", "comparison_position", "similarity", "total_minutes"]
        + [c for c in ("player_name", "name", "team_id") if c in pool]
    ]


def main():
    season = pd.read_parquet(DATA_DIR / "historical_player_season_stats.parquet")
    profiles = build_profiles(season)
    profiles.to_parquet(DATA_DIR / "player_similarity_profiles.parquet", index=False)
    results = []
    for row in profiles.itertuples(index=False):
        if row.primary_position == "GK":
            continue
        try:
            results.append(rank_similar_players(profiles, row.player_id))
        except ValueError as exc:
            print(f"Skipping {row.player_id}: {exc}")
    if not results:
        raise ValueError("No player similarity rankings were produced")
    ranking = pd.concat(results, ignore_index=True)
    ranking.to_parquet(DATA_DIR / "player_similarity_top10.parquet", index=False)
    print(f"Profiles: {len(profiles):,}; similarity pairs: {len(ranking):,}")
    print(ranking.head(10).to_string(index=False))
    print("Saved outputs to", DATA_DIR)


if __name__ == "__main__":
    main()
