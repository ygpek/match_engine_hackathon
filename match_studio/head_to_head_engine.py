"""Phase-safe, descriptive match-up comparisons for opposing starting players.

Historical profiles exclude the featured match. In-play shot contributions are
attributed only through observed event player IDs. Non-shot individual match
metrics are shown only at full-time, from the final player-stat dataset.
"""
from __future__ import annotations
import pandas as pd

BATTLES = (
    ('Goalkeepers', ('GK',), ('GK',), ('passes_completed', 'recoveries')),
    ('Strikers', ('ST', 'CF'), ('ST', 'CF'), ('goals', 'xg', 'shots', 'shots_on_target')),
    ('Midfielders', ('CM', 'AMW'), ('CM', 'AMW'), ('assists', 'key_passes', 'progressive_passes', 'tackles')),
)
LABELS = {'goals':'Goals', 'xg':'xG', 'shots':'Shots', 'shots_on_target':'Shots on target',
          'assists':'Assists', 'key_passes':'Key passes', 'progressive_passes':'Progressive passes',
          'tackles':'Tackles', 'passes_completed':'Passes completed', 'recoveries':'Recoveries',
          'red_cards':'Red cards', 'yellow_cards':'Yellow cards'}


def _number(value):
    try:
        result = float(value)
        return result if pd.notna(result) else 0.
    except (ValueError, TypeError):
        return 0.


def historical_rates(history, pid, last_n=10):
    rows = history.loc[history.player_id.astype(str).eq(str(pid))].copy()
    if '_match_date' in rows.columns:
        rows = rows.sort_values(['_match_date', 'match_id']).tail(last_n)
    else:
        rows = rows.tail(last_n)
    mins = sum(_number(v) for v in rows.get('minutes_played', []))
    result = {}
    if mins:
        for metric in LABELS:
            if metric in rows.columns:
                result[metric] = round(sum(_number(v) for v in rows[metric]) * 90 / mins, 2)
    return result, len(rows)


def select_battles(lineups, fixture, positions, history, max_battles=3):
    """Select GK, ST and CM match-ups from actual starters; stable across phases."""
    match_id = str(fixture['match_id'])
    if 'match_id' in lineups.columns:
        lineups = lineups.loc[lineups.match_id.astype(str).eq(match_id)]
    if 'started' in lineups.columns:
        lineups = lineups.loc[lineups.started.astype(bool)]
    pairs = []
    for title, home_pos, away_pos, metrics in BATTLES:
        chosen = []
        for side, eligible in (('home', home_pos), ('away', away_pos)):
            tid = str(fixture[f'{side}_team_id'])
            candidates = lineups.loc[lineups.team_id.astype(str).eq(tid), 'player_id'].astype(str).tolist()
            candidates = [p for p in candidates if positions.get(p) in eligible]
            if not candidates:
                break
            def rank(pid):
                rates, sample = historical_rates(history, pid)
                # Role-specific selection based on past contribution, not future match stats.
                if title == 'Strikers':
                    score = rates.get('goals', 0) * 3 + rates.get('xg', 0)
                elif title == 'Midfielders':
                    score = rates.get('key_passes', 0) + rates.get('assists', 0) * 2
                else:
                    score = sample
                return (-score, -sample, pid)
            chosen.append(sorted(candidates, key=rank)[0])
        if len(chosen) == 2:
            pairs.append({'title':title, 'home_player_id':chosen[0],
                          'away_player_id':chosen[1], 'historical_metrics':metrics})
        if len(pairs) >= max_battles:
            break
    return pairs


def observed_shot_stats(events, player_id, minute):
    stats = {'goals':0, 'xg':0., 'shots':0, 'shots_on_target':0, 'assists':0}
    for event in events:
        if int(event.get('minute') or 0) > minute:
            continue
        typ = event.get('event_type')
        if typ not in ('shot','goal'):
            continue
        if str(event.get('player_id')) == str(player_id):
            stats['shots'] += 1
            stats['shots_on_target'] += int(bool(event.get('is_on_target')) or typ == 'goal')
            stats['goals'] += int(typ == 'goal')
            stats['xg'] += _number(event.get('xg'))
        if typ == 'goal' and str(event.get('assist_player_id')) == str(player_id):
            stats['assists'] += 1
    stats['xg'] = round(stats['xg'],2)
    return stats


def final_stats(frame, fixture, player_id):
    rows = frame.loc[frame.player_id.astype(str).eq(str(player_id))]
    if 'match_id' in rows.columns:
        rows = rows.loc[rows.match_id.astype(str).eq(str(fixture['match_id']))]
    if rows.empty:
        return {}
    row = rows.iloc[0]
    return {metric:round(_number(row[metric]),2) for metric in LABELS if metric in row.index}
