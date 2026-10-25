"""Rank observed player profiles for a tactical need (NOT player similarity).

Place in engines/ alongside player_similarity_engine.py.
Input: data_generator/data/player_similarity_profiles.parquet
Example: python tactical_suitability_engine.py --team-id T01 --position CM --scenario protect_lead
        python tactical_suitability_engine.py --team-id T01 --position AMW --scenario chase_goal
Use --list-teams to inspect team IDs. Exports a CSV in engines/data/.
"""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data_generator" / "data"
OUTPUT_DIR = Path(__file__).resolve().parent / "data"
MIN_TOTAL_MINUTES = 900

# Positive metrics indicate more desirable performance; dispossessions are inverted.
FEATURE_GROUPS = {
    "attacking": {"xg_p90": 1.0, "goals_p90": 0.45, "xa_p90": 0.75, "key_passes_p90": 0.7},
    "passing": {"progressive_passes_p90": 1.0, "passes_final_third_p90": 0.7, "pass_completion_pct": 0.75},
    "defending": {"tackles_p90": 1.0, "interceptions_p90": 1.0, "recoveries_p90": 0.65, "clearances_p90": 0.45},
    "possession": {"progressive_carries_p90": 1.0, "successful_dribbles_p90": 0.7, "dispossessions_p90": -0.6},
    "physical": {"distance_km_p90": 0.7, "sprints_p90": 1.0, "accelerations_p90": 0.7},
}
SCENARIOS = {
    "balanced": {"attacking": 1, "passing": 1, "defending": 1, "possession": 1, "physical": 1},
    "chase_goal": {"attacking": 3, "passing": 1.5, "defending": 0.2, "possession": 1.7, "physical": 0.8},
    "protect_lead": {"attacking": 0.2, "passing": 1.4, "defending": 3, "possession": 1.7, "physical": 1},
    "control_midfield": {"attacking": 0.5, "passing": 3, "defending": 1, "possession": 2.5, "physical": 0.7},
    "high_press": {"attacking": 0.7, "passing": 0.5, "defending": 2.3, "possession": 0.5, "physical": 3},
    "counterattack": {"attacking": 2.5, "passing": 0.5, "defending": 0.6, "possession": 2.2, "physical": 2},
}


def load_profiles(path=DATA_DIR / "player_similarity_profiles.parquet"):
    profiles = pd.read_parquet(path)
    required = {"player_id", "team_id", "primary_position", "total_minutes"}
    missing = required - set(profiles.columns)
    if missing:
        raise ValueError(f"Missing profile columns: {sorted(missing)}")
    return profiles


def rank_suitability(
    profiles, team_id, position, scenario="balanced", candidate_ids=None, excluded_ids=None, weights=None, top_n=10
):
    """Rank same-team candidates against same-position league-wide reference distribution.

    candidate_ids: optional IDs of players currently on bench (strongly recommended
                   for real match decisions). Without it, ranks all eligible squad members.
    excluded_ids: optional unavailable/on-pitch IDs to remove.
    Scores are position-relative 0-100 percentiles, not probabilities or expected goals.
    """
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario {scenario!r}; choices: {list(SCENARIOS)}")
    weights = SCENARIOS[scenario] if weights is None else weights
    if set(weights) - set(FEATURE_GROUPS) or any(float(v) < 0 for v in weights.values()):
        raise ValueError("Weights must be nonnegative and use known feature groups")
    if sum(float(v) for v in weights.values()) <= 0:
        raise ValueError("At least one weight must be positive")
    pool = profiles.loc[(profiles.primary_position == position) & (profiles.total_minutes >= MIN_TOTAL_MINUTES)].copy()
    if position == "GK":
        raise ValueError("Goalkeepers require specialized metrics; not yet supported")
    if len(pool) < 5:
        raise ValueError(f"Not enough eligible {position} players for comparison")
    features = [f for group in FEATURE_GROUPS.values() for f in group]
    missing = set(features) - set(pool.columns)
    if missing:
        raise ValueError(f"Missing metrics: {sorted(missing)}")

    # Compare to players in the same position, not to all positions combined.
    # Percentile ranks are robust to wildly different scales across metrics.
    scores = pd.DataFrame(index=pool.index)
    for group_name, group in FEATURE_GROUPS.items():
        numerator = pd.Series(0.0, index=pool.index)
        denominator = 0.0
        for feature, importance in group.items():
            x = pd.to_numeric(pool[feature], errors="coerce").replace([np.inf, -np.inf], np.nan)
            if x.notna().sum() < 3 or x.nunique() < 2:
                continue
            x = x.fillna(x.median())
            pct = x.rank(pct=True, method="average") * 100
            if importance < 0:
                pct = 100 - pct
            numerator += pct * abs(importance)
            denominator += abs(importance)
        scores[group_name] = numerator / denominator if denominator else 50.0
    total_weight = sum(float(weights.get(g, 0)) for g in FEATURE_GROUPS)
    pool["suitability_score"] = sum(scores[g] * float(weights.get(g, 0)) for g in FEATURE_GROUPS) / total_weight
    for group in FEATURE_GROUPS:
        pool[f"{group}_score"] = scores[group]
    candidates = pool.loc[pool.team_id.astype(str) == str(team_id)].copy()
    if candidate_ids is not None:
        ids = {str(x) for x in candidate_ids}
        candidates = candidates.loc[candidates.player_id.astype(str).isin(ids)]
    if excluded_ids is not None:
        ids = {str(x) for x in excluded_ids}
        candidates = candidates.loc[~candidates.player_id.astype(str).isin(ids)]
    candidates = candidates.sort_values(["suitability_score", "total_minutes"], ascending=[False, False]).head(top_n)
    candidates.insert(0, "scenario", scenario)
    display = ["scenario", "player_id", "team_id", "primary_position", "suitability_score"]
    display += [f"{g}_score" for g in FEATURE_GROUPS]
    display += ["total_minutes"]
    display += [c for c in ("player_name", "name", "first_name", "last_name") if c in candidates]
    return candidates[display].reset_index(drop=True).round(2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--team-id", help="Team ID, e.g. T01")
    parser.add_argument("--position", choices=["CB", "WB", "CM", "AMW", "ST"])
    parser.add_argument("--scenario", choices=SCENARIOS, default="balanced")
    parser.add_argument("--candidate-ids", nargs="+", help="Restrict to bench player IDs")
    parser.add_argument("--exclude-ids", nargs="+", help="Exclude unavailable or on-pitch players")
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--list-teams", action="store_true")
    args = parser.parse_args()
    profiles = load_profiles()
    if args.list_teams:
        print("Team IDs:", ", ".join(sorted(profiles.team_id.astype(str).unique())))
        return
    if not args.team_id or not args.position:
        parser.error("--team-id and --position are required (or use --list-teams)")
    ranked = rank_suitability(
        profiles,
        team_id=args.team_id,
        position=args.position,
        scenario=args.scenario,
        candidate_ids=args.candidate_ids,
        excluded_ids=args.exclude_ids,
        top_n=args.top_n,
    )
    if ranked.empty:
        print("No eligible candidates found; check team, position and player filters.")
        return
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dest = OUTPUT_DIR / f"suitability_{args.team_id}_{args.position}_{args.scenario}.csv"
    ranked.to_csv(dest, index=False)
    print(ranked.to_string(index=False))
    print(f"\nSaved {dest}")
    print("Scores are position-relative performance percentiles, NOT probabilities.")


if __name__ == "__main__":
    main()
