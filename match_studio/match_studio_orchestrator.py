"""Match Studio: assemble four phase-safe, evidence-grounded analytics payloads.

Project layout:
  data_generator/data/*.parquet
  featured_match/data/featured_match*.{json,parquet}
  similarity_engine/{player_similarity_engine,tactical_suitability_engine}.py
  historical_context_engine/historical_context_engine.py
  match_studio/match_studio_orchestrator.py

Run from project root: python match_studio/match_studio_orchestrator.py
Optional: --phase prematch|halftime|live|postmatch --scenario protect_lead

No Azure/OpenAI calls: JSON is ready for a later narrative service.
Historical evidence and player profiles use ONLY matches dated before the
featured fixture, excluding its match ID regardless of date.
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from league_expectations import build_league_expectations
from player_performance_engine import evaluate_player_match

ROOT = Path(__file__).resolve().parent.parent
HIST_DATA = ROOT / "data_generator" / "data"
FEATURED_DATA = ROOT / "featured_match" / "data"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
PHASES = ("prematch", "halftime", "live", "postmatch")


def load_module(name, path):
    if not path.is_file():
        raise FileNotFoundError(f"Missing module: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(x) for x in obj]
    if isinstance(obj, np.ndarray):
        return [json_safe(x) for x in obj.tolist()]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    if obj is pd.NaT or obj is None:
        return None
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def read_parquet(path):
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}. Run data generators first.")
    return pd.read_parquet(path)


def cutoff_data(matches, featured):
    date = pd.Timestamp(featured["date"])
    mask = (pd.to_datetime(matches["date"]) < date) & (matches["match_id"].astype(str) != str(featured["match_id"]))
    eligible = matches.loc[mask].copy()
    return eligible, date


def build_asof_profiles(player_match_stats, matches, players, sim_engine):
    """Rebuild observed player profiles with no future or featured-match records."""
    from_feature = matches[["match_id", "date"]].copy()
    eligible_ids = set(from_feature.match_id.astype(str))
    safe_stats = player_match_stats[player_match_stats.match_id.astype(str).isin(eligible_ids)].copy()
    if safe_stats.empty:
        return pd.DataFrame(), 0

    # Import aggregator directly: no generated hidden/ability ratings are read.
    aggregator = load_module(
        "season_aggregator", ROOT / "data_generator" / "generate_historical_player_season_stats.py"
    )
    season = aggregator.aggregate(safe_stats, ["season_id"])
    ids = [
        c for c in ("player_id", "team_id", "player_name", "name", "primary_position", "age") if c in players.columns
    ]
    season = season.merge(players[ids], on="player_id", how="left", validate="many_to_one")
    if season.season_id.nunique() >= 2:
        try:
            return sim_engine.build_profiles(season), len(safe_stats)
        except ValueError:
            pass
    # Early-date fallback: only use seasons already played, never future values.
    features = list(dict.fromkeys(f for group in sim_engine.FEATURE_GROUPS.values() for f in group))
    eligible = season[season.minutes_played >= sim_engine.MIN_SEASON_MINUTES].copy()
    rows = []
    for pid, group in eligible.groupby("player_id"):
        total = int(group.minutes_played.sum())
        if total < sim_engine.MIN_TOTAL_MINUTES:
            continue
        row = {"player_id": pid, "total_minutes": total, "primary_position": group.primary_position.iloc[-1]}
        for col in ("team_id", "player_name", "name", "age"):
            if col in group:
                row[col] = group[col].iloc[-1]
        for feature in features:
            vals = pd.to_numeric(group[feature], errors="coerce")
            good = vals.notna()
            row[feature] = (
                float(np.average(vals[good], weights=group.loc[good, "minutes_played"])) if good.any() else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows), len(safe_stats)


def team_scenarios(snapshot, override):
    if override:
        return {"home": override, "away": override}
    home, away = snapshot["home_goals"], snapshot["away_goals"]
    if snapshot["phase"] == "prematch":
        return {"home": "balanced", "away": "balanced"}
    return {
        "home": "chase_goal" if home < away else "protect_lead" if home > away else "balanced",
        "away": "chase_goal" if away < home else "protect_lead" if away > home else "balanced",
    }


def player_recommendations(
    snapshot, metadata, lineups, players, profiles, sim_engine, suitability, top_n, scenario_override=None
):
    result = {}
    participant = lineups[["team_id", "player_id", "started", "minute_on", "minute_off"]].copy()
    for side, team_id in [("home", metadata["home_team_id"]), ("away", metadata["away_team_id"])]:
        squad = participant[participant.team_id.astype(str) == str(team_id)]
        active = set(str(x) for x in snapshot["active_player_ids"].get(str(team_id), []))
        # The historical_lineups file lists only players who actually participated.
        # Using its substitution rows as the bench would reveal future substitutions.
        # Instead, construct the bench from the full 22-player team roster.
        starters = set(squad.loc[squad.started, "player_id"].astype(str))
        appeared = set(squad.loc[(squad.minute_on <= snapshot["minute"]) & (~squad.started), "player_id"].astype(str))
        all_team_ids = set(players.loc[players.team_id.astype(str) == str(team_id), "player_id"].astype(str))
        candidate_ids = all_team_ids - starters - appeared - active
        active_rows = players[players.player_id.astype(str).isin(active)]
        bench_rows = players[players.player_id.astype(str).isin(candidate_ids)]
        roster = []
        scenarios = team_scenarios(snapshot, scenario_override)
        for position in ["CB", "WB", "CM", "AMW", "ST"]:
            pos_ids = bench_rows.loc[bench_rows.primary_position == position, "player_id"].astype(str).tolist()
            if not pos_ids:
                continue
            item = {"position": position, "eligible_bench_player_ids": pos_ids}
            if not profiles.empty:
                try:
                    ranked = suitability.rank_suitability(
                        profiles, str(team_id), position, scenario=scenarios[side], candidate_ids=pos_ids, top_n=top_n
                    )
                    item["ranked_candidates"] = json_safe(ranked.to_dict(orient="records"))
                except ValueError as exc:
                    item["ranking_unavailable_reason"] = str(exc)
            else:
                item["ranking_unavailable_reason"] = "Insufficient pre-match playing-time history"
            roster.append(item)
        # Compare bench candidates with currently active players in the SAME
        # position, using the SAME pre-match profiles and scenario-relative scale.
        # This is a suitability difference, not a predicted match outcome.
        substitution_pairs = []
        if not profiles.empty:
            for group in roster:
                position = group["position"]
                candidates = group.get("ranked_candidates", [])
                on_pitch_ids = (
                    active_rows.loc[active_rows.primary_position == position, "player_id"].astype(str).tolist()
                )
                if not candidates or not on_pitch_ids:
                    continue
                try:
                    incumbents = suitability.rank_suitability(
                        profiles,
                        str(team_id),
                        position,
                        scenario=scenarios[side],
                        candidate_ids=on_pitch_ids,
                        top_n=len(on_pitch_ids),
                    )
                except ValueError:
                    continue
                for candidate in candidates:
                    for _, incumbent in incumbents.iterrows():
                        difference = round(
                            float(candidate["suitability_score"]) - float(incumbent["suitability_score"]), 2
                        )
                        if difference <= 0:
                            continue
                        substitution_pairs.append(
                            {
                                "out_player_id": str(incumbent["player_id"]),
                                "in_player_id": str(candidate["player_id"]),
                                "out_player_name": str(
                                    players.loc[
                                        players.player_id.astype(str) == str(incumbent["player_id"]), "player_name"
                                    ].iloc[0]
                                ),
                                "in_player_name": str(
                                    players.loc[
                                        players.player_id.astype(str) == str(candidate["player_id"]), "player_name"
                                    ].iloc[0]
                                ),
                                "position": position,
                                "scenario": scenarios[side],
                                "out_suitability": round(float(incumbent["suitability_score"]), 2),
                                "in_suitability": round(float(candidate["suitability_score"]), 2),
                                "suitability_difference": difference,
                                "interpretation": "Higher historical scenario-relative suitability; "
                                "not a predicted in-match performance gain.",
                            }
                        )
        substitution_pairs.sort(key=lambda x: x["suitability_difference"], reverse=True)
        comparisons = []
        if not profiles.empty:
            for p in active_rows.itertuples(index=False):
                if p.primary_position == "GK":
                    continue
                if str(p.player_id) not in set(profiles.player_id.astype(str)):
                    continue
                try:
                    similar = sim_engine.rank_similar_players(profiles, p.player_id, top_n=top_n)
                    comparisons.append(
                        {
                            "starter_player_id": str(p.player_id),
                            "similar_players": json_safe(similar.to_dict(orient="records")),
                        }
                    )
                except ValueError:
                    continue
        result[side] = {
            "team_id": team_id,
            "active_player_ids": sorted(active),
            "available_bench_player_ids": sorted(candidate_ids),
            "scenario": scenarios[side],
            "substitution_options_by_position": roster,
            "starter_similarity": comparisons,
            "substitution_comparisons": substitution_pairs[:20],
            "note": "Rankings are historical position-relative scores, not win probabilities. "
            "Position-for-position options only; goalkeeper suggestions excluded.",
        }
    return result


def assemble(
    phase,
    metadata,
    snapshots,
    matches,
    all_matches,
    teams,
    team_profiles,
    team_stats,
    player_match_stats,
    featured_player_stats,
    players,
    lineups,
    events,
    context_engine,
    sim_engine,
    suitability,
    profiles,
    history_date,
    recent_n,
    top_n,
    scenario,
):
    snap = next((s for s in snapshots if s["phase"] == phase), None)
    if snap is None:
        raise ValueError(f"Missing phase snapshot {phase}")
    home, away = str(metadata["home_team_id"]), str(metadata["away_team_id"])
    # Historical engine sees already-filtered matches, so featured/future results cannot leak.
    context = context_engine.build_context(
        matches,
        teams,
        team_stats,
        team_id=home,
        opponent_id=away,
        phase=phase,
        home_goals=None if phase == "prematch" else snap["home_goals"],
        away_goals=None if phase == "prematch" else snap["away_goals"],
        as_of=None,
        recent_n=recent_n,
    )
    context["as_of_exclusive"] = str(history_date.date())
    context["excluded_match_id"] = str(metadata["match_id"])
    context["historical_matches_available"] = int(len(matches))

    # Authoritative team-specific form for AI; preserve legacy historical_context
    # for Streamlit and existing consumers. "away_form" in historical_context
    # is the HOME team's away-venue split, not the opponent's overall form.
    team_recent_form = {
        "home": {"team_id": home, "recent_form": context.get("recent_form")},
        "away": {"team_id": away, "recent_form": context.get("opponent_recent_form")},
    }
    context["form_field_semantics"] = {
        "recent_form": "Featured home team's overall recent matches",
        "home_form": "Featured home team's matches played at home only",
        "away_form": "Featured home team's matches played away only; NOT featured away team's form",
        "opponent_recent_form": "Featured away team's overall recent matches",
    }

    # Only public fixture metadata; ht_score / ft_score are explicitly not copied.
    fixture = {k: metadata[k] for k in ("match_id", "date", "home_team_id", "away_team_id")}
    scenario_by_side = team_scenarios(snap, scenario)
    recommendations = player_recommendations(
        snap, metadata, lineups, players, profiles, sim_engine, suitability, top_n, scenario
    )
    for side in ("home", "away"):
        recommendations[side]["scenario"] = scenario_by_side[side]
    time = int(snap["minute"])
    visible = events[(events.minute < time) | ((events.minute == time) & (events.second <= 59))]
    # Exclude halftime/fulltime events at prematch and prevent event-log future disclosures.
    if phase == "prematch":
        visible = visible.iloc[0:0]
    event_cols = ["minute", "second", "team_id", "event_type", "player_id", "description", "xg", "is_goal"]
    event_cols = [c for c in event_cols if c in visible.columns]
    league_expectations = (
        build_league_expectations(all_matches, teams, team_profiles, metadata, snap, phase)
        if phase in ("prematch", "postmatch") else None
    )
    individual_review = (
        evaluate_player_match(featured_player_stats, players, fixture=metadata, lineups=lineups)
        if phase == "postmatch" else None
    )
    return json_safe(
        {
            "individual_performance_review": individual_review,
            "schema_version": "1.0",
            "phase": phase,
            "fixture": fixture,
            "state": snap,
            "league_expectations": league_expectations,
            "historical_context": context,
            "team_recent_form": team_recent_form,
            "tactical_analysis": recommendations,
            "observed_events": visible[event_cols].to_dict(orient="records"),
            "evidence_provenance": {
                "historical_cutoff_exclusive": str(history_date.date()),
                "excluded_match_id": str(metadata["match_id"]),
                "historical_match_count": int(len(matches)),
                "player_profiles_recomputed_before_cutoff": True,
                "phase_data_policy": "Only snapshot/event observations through current minute; "
                "passing and defending totals only from completed halves.",
            },
            "ai_instructions": {
                "grounding": "Use only this JSON. Never claim events not present in observed_events.",
                "uncertainty": "Treat historical rates as descriptive, not forecasts.",
                "recent_form_mapping": (
                    "For comparisons of the two featured teams, exclusively use "
                    "team_recent_form.home.recent_form for the home team and "
                    "team_recent_form.away.recent_form for the away team. "
                    "historical_context.away_form is the home team's away-venue split, "
                    "NOT the featured away team's form."
                ),
                "narrative_focus": {
                    "prematch": "Preview team form, starting XI and available tactical options; do not mention score.",
                    "halftime": "Explain halftime observations, historical analogues and substitution options.",
                    "live": "Discuss score and shot momentum known through the snapshot minute; no future events.",
                    "postmatch": "Summarize observed result, full-match stats, league context, and individual_performance_review awards with player names and concrete statistics.",
                }[phase],
            },
        }
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=PHASES, help="Single phase; default all four")
    parser.add_argument(
        "--scenario",
        choices=["balanced", "chase_goal", "protect_lead", "control_midfield", "high_press", "counterattack"],
    )
    parser.add_argument("--recent-n", type=int, default=5)
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    if args.recent_n < 1 or args.top_n < 1:
        parser.error("--recent-n and --top-n must be positive")

    context_engine = load_module("context_engine", ROOT / "engines" / "historical_context_engine.py")
    sim_engine = load_module("sim_engine", ROOT / "engines" / "player_similarity_engine.py")
    suitability = load_module("suitability_engine", ROOT / "engines" / "tactical_suitability_engine.py")
    metadata = json.loads((FEATURED_DATA / "featured_match.json").read_text(encoding="utf-8"))
    snapshots = json.loads((FEATURED_DATA / "featured_match_snapshots.json").read_text(encoding="utf-8"))
    matches = read_parquet(HIST_DATA / "historical_matches.parquet")
    teams = read_parquet(HIST_DATA / "teams.parquet")
    team_stats = read_parquet(HIST_DATA / "historical_team_stats.parquet")
    player_match_stats = read_parquet(HIST_DATA / "historical_player_match_stats.parquet")
    featured_player_stats = read_parquet(FEATURED_DATA / "featured_match_player_stats.parquet")
    players = read_parquet(HIST_DATA / "players.parquet")
    lineups = read_parquet(FEATURED_DATA / "featured_match_lineups.parquet")
    events = read_parquet(FEATURED_DATA / "featured_match_events.parquet")
    eligible_matches, cutoff = cutoff_data(matches, metadata)
    if eligible_matches.empty:
        print("WARNING: No historical matches predate the featured fixture; historical context will be empty.")
    team_profiles = read_parquet(HIST_DATA/'team_profiles.parquet')
    profiles, appearance_count = build_asof_profiles(player_match_stats, eligible_matches, players, sim_engine)
    print(
        f"Historical cutoff: {cutoff.date()}, pre-match records: {len(eligible_matches)}, "
        f"player appearances: {appearance_count}, eligible player profiles: {len(profiles)}"
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for phase in ([args.phase] if args.phase else PHASES):
        payload = assemble(
            phase,
            metadata,
            snapshots,
            eligible_matches,
            matches,
            teams,
            team_profiles,
            team_stats,
            player_match_stats,
            featured_player_stats,
            players,
            lineups,
            events,
            context_engine,
            sim_engine,
            suitability,
            profiles,
            cutoff,
            args.recent_n,
            args.top_n,
            args.scenario,
        )
        dest = args.output_dir / f"match_studio_{phase}.json"
        dest.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        print(f"Saved {dest}")


if __name__ == "__main__":
    main()
