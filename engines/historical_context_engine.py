"""Historical evidence for Match Studio AI, grounded in generated match and team stats.

Example:
  python historical_context_engine.py --team-id T001 --opponent-id T002 --phase halftime --home-goals 1 --away-goals 0 --as-of 2026-10-01

Requires pandas, pyarrow. Reads ../data_generator/data by default.
Outputs JSON to stdout, or to --output path. No generative AI is used here.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / 'data_generator' / 'data'


def load_data(data_dir):
    matches = pd.read_parquet(data_dir / 'historical_matches.parquet')
    teams = pd.read_parquet(data_dir / 'teams.parquet')
    stats = pd.read_parquet(data_dir / 'historical_team_stats.parquet')
    matches['date'] = pd.to_datetime(matches['date'])
    if matches.match_id.duplicated().any():
        raise ValueError('Duplicate match IDs')
    if stats.duplicated(['match_id', 'team_id', 'period']).any():
        raise ValueError('Duplicate team-half statistics')
    return matches, teams, stats


def team_match_view(matches, stats):
    numeric = ['shots', 'shots_on_target', 'xg', 'goals', 'passes_attempted',
               'passes_completed', 'progressive_passes', 'tackles', 'interceptions',
               'recoveries', 'corners']
    numeric = [c for c in numeric if c in stats.columns]
    agg = stats.groupby(['match_id', 'team_id'], as_index=False)[numeric].sum()
    possession = stats.groupby(['match_id', 'team_id'], as_index=False).agg(
        possession_seconds=('possession_seconds', 'sum'),
        in_play_seconds=('in_play_seconds', 'sum'))
    agg = agg.merge(possession, on=['match_id', 'team_id'], validate='one_to_one')
    agg['possession_pct'] = 100 * agg.possession_seconds / agg.in_play_seconds
    home = matches[['match_id', 'season_id', 'date', 'home_team_id', 'away_team_id',
                    'ht_home_goals', 'ht_away_goals', 'ft_home_goals', 'ft_away_goals']].copy()
    h = home.assign(team_id=home.home_team_id, opponent_id=home.away_team_id,
                    venue='home', ht_for=home.ht_home_goals, ht_against=home.ht_away_goals,
                    goals_for=home.ft_home_goals, goals_against=home.ft_away_goals)
    a = home.assign(team_id=home.away_team_id, opponent_id=home.home_team_id,
                    venue='away', ht_for=home.ht_away_goals, ht_against=home.ht_home_goals,
                    goals_for=home.ft_away_goals, goals_against=home.ft_home_goals)
    view = pd.concat([h, a], ignore_index=True).merge(
        agg, on=['match_id', 'team_id'], how='left', validate='one_to_one')
    if view[numeric].isna().any().any():
        raise ValueError('Some historical team-match statistics are missing')
    view['points'] = ((view.goals_for > view.goals_against).astype(int) * 3
                      + (view.goals_for == view.goals_against).astype(int))
    view['result'] = view.apply(lambda r: 'W' if r.goals_for > r.goals_against else
                                'L' if r.goals_for < r.goals_against else 'D', axis=1)
    return view.sort_values(['date', 'match_id'])


def round_float(value, digits=2):
    return round(float(value), digits)


def recent_form(rows, n):
    sample = rows.tail(n)
    if sample.empty:
        return None
    return {
        'sample_size': len(sample),
        'record_oldest_to_newest': ''.join(sample.result),
        'wins': int((sample.result == 'W').sum()),
        'draws': int((sample.result == 'D').sum()),
        'losses': int((sample.result == 'L').sum()),
        'points': int(sample.points.sum()),
        'goals_for': int(sample.goals_for.sum()),
        'goals_against': int(sample.goals_against.sum()),
        'xg_for_per_match': round_float(sample.xg.mean()),
        'shots_per_match': round_float(sample.shots.mean()),
        'possession_pct_mean': round_float(sample.possession_pct.mean()),
        'source_match_ids': sample.match_id.tolist(),
    }


def halftime_history(rows, condition, n):
    if condition == 'leading':
        sample = rows[rows.ht_for > rows.ht_against].tail(n)
    elif condition == 'trailing':
        sample = rows[rows.ht_for < rows.ht_against].tail(n)
    else:
        sample = rows[rows.ht_for == rows.ht_against].tail(n)
    if sample.empty:
        return None
    return {
        'halftime_state': condition,
        'sample_size': len(sample),
        'full_time_wins': int((sample.result == 'W').sum()),
        'full_time_draws': int((sample.result == 'D').sum()),
        'full_time_losses': int((sample.result == 'L').sum()),
        'full_time_win_rate_pct': round_float(100 * (sample.result == 'W').mean(), 1),
        'source_match_ids': sample.match_id.tolist(),
    }


def build_context(matches, teams, stats, team_id, opponent_id=None,
                  phase='prematch', home_goals=None, away_goals=None,
                  as_of=None, recent_n=5, condition_n=20):
    ids = set(teams.team_id)
    if team_id not in ids or (opponent_id is not None and opponent_id not in ids):
        raise ValueError('Unknown team_id or opponent_id')
    if opponent_id == team_id:
        raise ValueError('A team cannot play itself')
    if phase not in {'prematch', 'halftime', 'live', 'postmatch'}:
        raise ValueError('Invalid phase')
    if (home_goals is None) != (away_goals is None):
        raise ValueError('Provide both --home-goals and --away-goals, or neither')
    if home_goals is not None and (home_goals < 0 or away_goals < 0):
        raise ValueError('Scores cannot be negative')
    if recent_n < 1 or condition_n < 1:
        raise ValueError('Sample sizes must be positive')

    cutoff = pd.Timestamp(as_of) if as_of else pd.Timestamp.max
    # Strict cutoff prevents accidentally using future results as historical evidence.
    eligible = matches[matches.date < cutoff]
    view = team_match_view(eligible, stats)
    names = teams.set_index('team_id').team_name.to_dict()
    selected = view[view.team_id == team_id]
    result = {
        'phase': phase,
        'as_of_exclusive': str(cutoff.date()) if as_of else None,
        'team': {'id': team_id, 'name': names[team_id]},
        'opponent': {'id': opponent_id, 'name': names[opponent_id]} if opponent_id else None,
        'historical_matches_available': len(selected),
        'recent_form': recent_form(selected, recent_n),
        'home_form': recent_form(selected[selected.venue == 'home'], recent_n),
        'away_form': recent_form(selected[selected.venue == 'away'], recent_n),
        'halftime_patterns': {
            state: halftime_history(selected, state, condition_n)
            for state in ('leading', 'level', 'trailing')
        },
    }
    if opponent_id:
        other = view[view.team_id == opponent_id]
        h2h = selected[selected.opponent_id == opponent_id].tail(5)
        result['opponent_recent_form'] = recent_form(other, recent_n)
        result['opponent_halftime_patterns'] = {
            state: halftime_history(other, state, condition_n)
            for state in ('leading', 'level', 'trailing')
        }
        result['head_to_head'] = {
            'sample_size': len(h2h),
            'wins': int((h2h.result == 'W').sum()),
            'draws': int((h2h.result == 'D').sum()),
            'losses': int((h2h.result == 'L').sum()),
            'source_match_ids': h2h.match_id.tolist(),
        }
    if home_goals is not None:
        # Interpret score from the perspective of --team-id, which is the home team.
        state = ('leading' if home_goals > away_goals else
                 'trailing' if home_goals < away_goals else 'level')
        result['current_match_state'] = {
            'team_goals': home_goals, 'opponent_goals': away_goals,
            'score_state': state,
            'historical_comparison': result['halftime_patterns'][state]
            if phase == 'halftime' else None,
        }
    result['evidence_note'] = ('All aggregates come from historical synthetic matches. '
                               'Rates are descriptive, not predictions. '
                               'Source match IDs are included for traceability.')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument('--team-id', required=True)
    parser.add_argument('--opponent-id')
    parser.add_argument('--phase', choices=['prematch', 'halftime', 'live', 'postmatch'], default='prematch')
    parser.add_argument('--home-goals', type=int)
    parser.add_argument('--away-goals', type=int)
    parser.add_argument('--as-of', help='Only matches dated before YYYY-MM-DD are included')
    parser.add_argument('--recent-n', type=int, default=5)
    parser.add_argument('--condition-n', type=int, default=20)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    matches, teams, stats = load_data(args.data_dir)
    context = build_context(matches, teams, stats, args.team_id, args.opponent_id,
                            args.phase, args.home_goals, args.away_goals,
                            args.as_of, args.recent_n, args.condition_n)
    payload = json.dumps(context, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding='utf-8')
        print(f'Saved {args.output}')
    else:
        print(payload)


if __name__ == '__main__':
    main()
