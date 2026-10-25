"""Season-to-date player leaderboards, with featured-fixture phase safety."""
from pathlib import Path
import pandas as pd

DEFENSIVE_FIELDS = ('tackles', 'interceptions', 'recoveries', 'clearances')


def league_player_leaders(data_dir: Path, fixture: dict, phase: str, top_n: int = 5):
    data_dir = Path(data_dir)
    matches = pd.read_parquet(data_dir / 'historical_matches.parquet').copy()
    player_matches = pd.read_parquet(data_dir / 'historical_player_match_stats.parquet').copy()
    players = pd.read_parquet(data_dir / 'players.parquet').copy()
    matches['date'] = pd.to_datetime(matches['date'])
    featured_id = str(fixture['match_id'])
    featured = matches.loc[matches.match_id.astype(str) == featured_id]
    if featured.empty:
        raise ValueError('Featured fixture is missing from historical_matches.parquet')
    season = str(fixture.get('season_id') or featured.iloc[0]['season_id'])
    kickoff_date = pd.Timestamp(fixture['date']).normalize()
    eligible = matches.loc[
        (matches.season_id.astype(str) == season)
        & (matches.date.dt.normalize() < kickoff_date)
        & (matches.match_id.astype(str) != featured_id)
    ].copy()
    if phase == 'postmatch':
        eligible = pd.concat([eligible, featured], ignore_index=True)
    match_ids = set(eligible.match_id.astype(str))
    stats = player_matches.loc[
        player_matches.match_id.astype(str).isin(match_ids)
        & (player_matches.season_id.astype(str) == season)
    ].copy()
    names_col = 'player_name' if 'player_name' in players else 'name'
    identity = players[['player_id', 'team_id', names_col, 'primary_position']].copy()
    identity = identity.rename(columns={names_col: 'Player', 'primary_position': 'Position'})
    identity['player_id'] = identity.player_id.astype(str)
    identity['team_id'] = identity.team_id.astype(str)
    team_names = pd.read_parquet(data_dir / 'teams.parquet')
    clubs = dict(zip(team_names.team_id.astype(str), team_names.team_name.astype(str)))
    if stats.empty:
        return {k: pd.DataFrame() for k in ('goals', 'assists', 'defenders', 'keepers')}
    stats['player_id'] = stats.player_id.astype(str)
    stats['team_id'] = stats.team_id.astype(str)
    numeric = ['goals', 'assists', 'minutes_played', *DEFENSIVE_FIELDS]
    for col in numeric:
        stats[col] = pd.to_numeric(stats[col], errors='coerce').fillna(0)
    agg = stats.groupby('player_id', as_index=False).agg(
        Goals=('goals', 'sum'), Assists=('assists', 'sum'),
        Appearances=('match_id', 'nunique'), Minutes=('minutes_played', 'sum'),
        **{col: (col, 'sum') for col in DEFENSIVE_FIELDS}
    )
    agg = agg.merge(identity, on='player_id', how='left', validate='one_to_one')
    agg['Club'] = agg.team_id.map(clubs).fillna(agg.team_id)
    agg['Defensive actions'] = agg[list(DEFENSIVE_FIELDS)].sum(axis=1)
    agg['Actions / game'] = (agg['Defensive actions'] / agg['Appearances']).round(2)
    # A goalkeeper earns a clean sheet only for a full 90-minute appearance
    # in a fixture where their team conceded no goals.
    clean = eligible[['match_id', 'home_team_id', 'away_team_id', 'ft_home_goals', 'ft_away_goals']].copy()
    clean['match_id'] = clean.match_id.astype(str)
    gk_rows = stats.merge(identity[['player_id', 'Position']], on='player_id', how='left')
    gk_rows = gk_rows.loc[gk_rows.Position == 'GK'].merge(clean, on='match_id', how='left', validate='many_to_one')
    gk_rows['conceded'] = gk_rows.apply(
        lambda r: r.ft_away_goals if str(r.team_id) == str(r.home_team_id)
        else (r.ft_home_goals if str(r.team_id) == str(r.away_team_id) else float('nan')), axis=1
    )
    gk_rows['clean_sheet'] = ((gk_rows.minutes_played >= 90) & (gk_rows.conceded == 0)).astype(int)
    cs = gk_rows.groupby('player_id')['clean_sheet'].sum()
    agg['Clean sheets'] = agg.player_id.map(cs).fillna(0).astype(int)
    common = ['Player', 'Club', 'Position']
    goals = agg.sort_values(['Goals', 'Assists', 'Player'], ascending=[False, False, True]).head(top_n)
    assists = agg.sort_values(['Assists', 'Goals', 'Player'], ascending=[False, False, True]).head(top_n)
    defenders = agg.loc[agg.Position.isin(['CB', 'WB']) & (agg.Appearances >= 3)].sort_values(
        ['Actions / game', 'Defensive actions', 'Player'], ascending=[False, False, True]).head(top_n)
    keepers = agg.loc[agg.Position == 'GK'].sort_values(
        ['Clean sheets', 'Appearances', 'Player'], ascending=[False, False, True]).head(top_n)
    return {
        'goals': goals[common + ['Goals', 'Assists', 'Appearances']].reset_index(drop=True),
        'assists': assists[common + ['Assists', 'Goals', 'Appearances']].reset_index(drop=True),
        'defenders': defenders[common + ['Actions / game', 'Defensive actions', 'Appearances']].reset_index(drop=True),
        'keepers': keepers[common + ['Clean sheets', 'Appearances']].reset_index(drop=True),
    }
