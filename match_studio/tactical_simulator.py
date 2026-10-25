"""On-demand, phase-safe, same-position substitution comparisons.

Uses the orchestrator's pre-fixture profile builder and the existing tactical
suitability engine. Never uses future match data or makes an Azure request.
"""
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def load_pre_match_profiles(fixture):
    """Rebuild profiles strictly from historical matches before this fixture."""
    from match_studio_orchestrator import (
        load_module, read_parquet, cutoff_data, build_asof_profiles
    )
    hist = ROOT / 'data_generator' / 'data'
    matches = read_parquet(hist / 'historical_matches.parquet')
    players = read_parquet(hist / 'players.parquet')
    player_stats = read_parquet(hist / 'historical_player_match_stats.parquet')
    safe_matches, _ = cutoff_data(matches, fixture)
    sim = load_module('sim_engine_simulator', ROOT / 'engines' / 'player_similarity_engine.py')
    profiles, _ = build_asof_profiles(player_stats, safe_matches, players, sim)
    return profiles


def evaluate_substitution(profiles, payload, side, outgoing_id, incoming_id, scenario):
    """Return comparable scores for two eligible players; raise ValueError otherwise."""
    from match_studio_orchestrator import load_module
    engine = load_module('suitability_simulator', ROOT / 'engines' / 'tactical_suitability_engine.py')
    if scenario not in engine.SCENARIOS:
        raise ValueError('Unknown tactical scenario')
    if payload.get('phase') not in ('halftime', 'live'):
        raise ValueError('Simulator is available only at halftime and live')
    section = payload['tactical_analysis'][side]
    team_id = str(section['team_id'])
    active = {str(x) for x in section['active_player_ids']}
    bench = {str(x) for x in section['available_bench_player_ids']}
    outgoing_id, incoming_id = str(outgoing_id), str(incoming_id)
    if outgoing_id not in active or incoming_id not in bench:
        raise ValueError('Selected players are not eligible for this phase')
    if profiles.empty:
        raise ValueError('No pre-match profiles are available')
    selected = profiles[profiles.player_id.astype(str).isin([outgoing_id, incoming_id])]
    if len(selected) != 2:
        raise ValueError('Insufficient historical playing time for one or both players')
    pos_out = str(selected.loc[selected.player_id.astype(str) == outgoing_id, 'primary_position'].iloc[0])
    pos_in = str(selected.loc[selected.player_id.astype(str) == incoming_id, 'primary_position'].iloc[0])
    if pos_out != pos_in or pos_out == 'GK':
        raise ValueError('Only same-position outfield replacements are supported')
    if not all(selected.team_id.astype(str) == team_id):
        raise ValueError('Players must belong to the selected team')
    # Critical: score BOTH players together on the same league-wide reference
    # distribution, using the exact original suitability function.
    ranked = engine.rank_suitability(
        profiles, team_id=team_id, position=pos_out, scenario=scenario,
        candidate_ids=[outgoing_id, incoming_id], top_n=2,
    )
    scores = {str(r.player_id): r for r in ranked.itertuples(index=False)}
    if outgoing_id not in scores or incoming_id not in scores:
        raise ValueError('A selected player does not meet the scoring eligibility threshold')
    old, new = scores[outgoing_id], scores[incoming_id]
    return {
        'position': pos_out, 'scenario': scenario,
        'out_score': float(old.suitability_score),
        'in_score': float(new.suitability_score),
        'difference': round(float(new.suitability_score) - float(old.suitability_score), 2),
        'categories': {
            category: (float(getattr(old, f'{category}_score')), float(getattr(new, f'{category}_score')))
            for category in engine.FEATURE_GROUPS
        },
    }
