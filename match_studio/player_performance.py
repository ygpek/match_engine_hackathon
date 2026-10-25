"""Phase-safe historical player profiles and observable match participation.

Historical metrics use only matches before the featured fixture. Match-level
individual actions are deliberately not inferred from anonymous shot events.
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
HIST = ROOT / 'data_generator' / 'data'
FEATURED = ROOT / 'featured_match' / 'data'
METRICS = {
    'goals': 'Goals', 'xg': 'Expected goals (xG)', 'shots': 'Shots',
    'assists': 'Assists', 'xa': 'Expected assists (xA)',
    'key_passes': 'Key passes', 'progressive_passes': 'Progressive passes',
    'tackles': 'Tackles', 'interceptions': 'Interceptions',
    'recoveries': 'Recoveries', 'progressive_carries': 'Progressive carries',
    'successful_dribbles': 'Successful dribbles', 'sprints': 'Sprints',
}


def load_player_history(fixture):
    matches = pd.read_parquet(HIST / 'historical_matches.parquet')
    stats = pd.read_parquet(HIST / 'historical_player_match_stats.parquet')
    eligible = matches.loc[
        (pd.to_datetime(matches.date) < pd.Timestamp(fixture['date'])) &
        (matches.match_id.astype(str) != str(fixture['match_id'])), 'match_id'
    ].astype(str)
    dates = matches[['match_id', 'date']].rename(columns={'date': '_match_date'})
    stats = stats[stats.match_id.astype(str).isin(set(eligible))].copy()
    return stats.merge(dates, on='match_id', how='left')


def player_profile(history, player_id, last_n=10):
    rows = history[history.player_id.astype(str) == str(player_id)].copy()
    if rows.empty:
        return {'matches': 0, 'minutes': 0, 'metrics': {}}
    # Most recent appearances, using date if available in caller's history.
    if '_match_date' in rows:
        rows = rows.sort_values(['_match_date', 'match_id']).tail(last_n)
    else:
        rows = rows.tail(last_n)
    minutes = float(pd.to_numeric(rows.minutes_played, errors='coerce').fillna(0).sum())
    metrics = {}
    for col, label in METRICS.items():
        if col in rows and minutes > 0:
            metrics[label] = round(float(pd.to_numeric(rows[col], errors='coerce').fillna(0).sum()) * 90 / minutes, 2)
    if 'passes_attempted' in rows and 'passes_completed' in rows:
        attempted = float(rows.passes_attempted.sum())
        if attempted:
            metrics['Pass completion %'] = round(100 * float(rows.passes_completed.sum()) / attempted, 1)
    return {'matches': len(rows), 'minutes': int(minutes), 'metrics': metrics}


def participation(lineups, player_id, minute):
    rows = lineups[lineups.player_id.astype(str) == str(player_id)]
    if rows.empty:
        return None
    row = rows.iloc[0]
    on, off = int(row.minute_on), int(row.minute_off)
    observed = max(0, min(int(minute), off) - on)
    return {'started': bool(row.started), 'minute_on': on if on <= minute else None,
            'minute_off': off if off <= minute else None, 'observed_minutes': observed,
            'currently_on_pitch': on <= minute < off}
