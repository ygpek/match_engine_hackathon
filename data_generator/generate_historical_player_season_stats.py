"""Aggregate observed player-match performances into season and overall profiles.

Run after generate_historical_player_match_stats.py.
Inputs: data/players.parquet, data/historical_player_match_stats.parquet
Outputs: data/historical_player_season_stats.parquet,
         data/historical_player_career_stats.parquet
No hidden player abilities are used in the aggregation.
"""
from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / 'data'
MIN_MINUTES_FOR_PCA = 450
COUNT_METRICS = [
    'shots', 'shots_on_target', 'goals', 'xg', 'assists', 'xa', 'key_passes',
    'passes_attempted', 'passes_completed', 'progressive_passes',
    'passes_final_third', 'tackles', 'interceptions', 'recoveries',
    'clearances', 'fouls', 'yellow_cards', 'red_cards',
    'progressive_carries', 'successful_dribbles', 'dispossessions',
    'distance_km', 'sprints', 'accelerations',
]
RATE_METRICS = COUNT_METRICS


def safe_pct(numerator, denominator):
    """NaN when the player has no opportunities; don't treat 0/0 as 0%."""
    return (100 * numerator / denominator.replace(0, np.nan)).round(2)


def aggregate(stats, group_cols):
    required = set(group_cols + ['player_id', 'match_id', 'minutes_played', 'started'] + COUNT_METRICS)
    missing = required - set(stats.columns)
    if missing:
        raise ValueError(f'Missing player-match columns: {sorted(missing)}')
    if stats.duplicated(['match_id', 'player_id']).any():
        raise ValueError('Duplicate player appearances: check player-match source')
    if (stats.minutes_played <= 0).any():
        raise ValueError('Minutes played must be positive')

    keys = ['player_id'] + group_cols
    aggregations = {
        'appearances': ('match_id', 'nunique'),
        'starts': ('started', 'sum'),
        'minutes_played': ('minutes_played', 'sum'),
    }
    aggregations.update({metric: (metric, 'sum') for metric in COUNT_METRICS})
    result = stats.groupby(keys, as_index=False, dropna=False).agg(**aggregations)
    result['minutes_per_appearance'] = (result.minutes_played / result.appearances).round(1)
    for metric in RATE_METRICS:
        result[f'{metric}_p90'] = (90 * result[metric] / result.minutes_played).round(3)
    result['pass_completion_pct'] = safe_pct(result.passes_completed, result.passes_attempted)
    result['shot_accuracy_pct'] = safe_pct(result.shots_on_target, result.shots)
    result['shot_conversion_pct'] = safe_pct(result.goals, result.shots)
    result['goals_minus_xg'] = (result.goals - result.xg).round(3)
    result['goals_minus_xg_p90'] = (90 * result.goals_minus_xg / result.minutes_played).round(3)
    result['eligible_for_pca'] = result.minutes_played >= MIN_MINUTES_FOR_PCA
    return result


def validate(season, career, match_stats):
    assert not season.duplicated(['player_id', 'season_id']).any()
    assert not career.duplicated(['player_id']).any()
    assert (season.starts <= season.appearances).all()
    assert (season.passes_completed <= season.passes_attempted).all()
    assert (season.progressive_passes <= season.passes_completed).all()
    assert (season.shots_on_target <= season.shots).all()
    assert (season.goals <= season.shots_on_target).all()
    assert season.minutes_played.sum() == match_stats.minutes_played.sum()
    for metric in COUNT_METRICS:
        assert np.isclose(season[metric].sum(), match_stats[metric].sum()), metric
        assert np.isclose(career[metric].sum(), match_stats[metric].sum()), metric


def main():
    stats = pd.read_parquet(DATA_DIR / 'historical_player_match_stats.parquet')
    players = pd.read_parquet(DATA_DIR / 'players.parquet')
    season = aggregate(stats, ['season_id'])
    career = aggregate(stats, [])
    # Public identity fields only; avoid exposing hidden simulation abilities.
    identity_cols = [c for c in ['player_id', 'team_id', 'player_name', 'name',
        'first_name', 'last_name', 'primary_position', 'secondary_position',
        'age', 'nationality'] if c in players.columns]
    if 'player_id' not in identity_cols:
        raise ValueError('players.parquet must contain player_id')
    if players.player_id.duplicated().any():
        raise ValueError('Duplicate player IDs in players.parquet')
    for frame in (season, career):
        frame_idx = frame.merge(players[identity_cols], on='player_id', how='left', validate='many_to_one')
        if len(frame_idx) != len(frame):
            raise ValueError('Identity join changed the row count')
        if frame is season:
            season = frame_idx
        else:
            career = frame_idx
    validate(season, career, stats)
    season.to_parquet(DATA_DIR / 'historical_player_season_stats.parquet', index=False)
    career.to_parquet(DATA_DIR / 'historical_player_career_stats.parquet', index=False)
    print(f'Season rows: {len(season):,}; career rows: {len(career):,}')
    print(f'PCA-eligible season rows (>={MIN_MINUTES_FOR_PCA} minutes): {season.eligible_for_pca.sum():,}')
    cols = ['player_id', 'season_id', 'minutes_played', 'goals_p90',
            'xg_p90', 'progressive_passes_p90', 'pass_completion_pct', 'tackles_p90']
    print(season[cols].head(8).to_string(index=False))
    print('Validation passed. Files saved in', DATA_DIR)


if __name__ == '__main__':
    main()
