"""Phase-safe league standings and 15-minute shot/goal interval summaries.

Only matches strictly before the featured fixture date contribute to history.
Post-match standings additionally include the featured match itself.
"""
from pathlib import Path
import pandas as pd

INTERVALS = [(0, 15), (15, 30), (30, 45), (45, 60), (60, 75), (75, 90)]


def load_history(data_dir: Path):
    matches = pd.read_parquet(data_dir / 'historical_matches.parquet')
    shots = pd.read_parquet(data_dir / 'historical_shots.parquet')
    teams = pd.read_parquet(data_dir / 'teams.parquet')
    matches['date'] = pd.to_datetime(matches['date'])
    return matches, shots, teams


def prior_matches(matches, fixture):
    date = pd.Timestamp(fixture['date']).normalize()
    return matches.loc[(matches['date'].dt.normalize() < date) &
                       (matches['match_id'].astype(str) != str(fixture['match_id']))].copy()


def league_table(matches, teams, fixture, phase):
    prior = prior_matches(matches, fixture)
    # League standings are season-specific; never mix seasons.
    match = matches.loc[matches.match_id.astype(str) == str(fixture['match_id'])]
    season = fixture.get('season_id') or (match.iloc[0]['season_id'] if not match.empty else None)
    if season is None:
        raise ValueError('Cannot identify featured match season for standings')
    prior = prior.loc[prior.season_id.astype(str) == str(season)]
    games = prior.copy()
    if phase == 'postmatch':
        if match.empty:
            raise ValueError('Featured match not found; cannot calculate post-match table')
        games = pd.concat([games, match], ignore_index=True)
    names = dict(zip(teams.team_id.astype(str), teams.team_name.astype(str)))
    rows = []
    for tid, name in names.items():
        home = games[games.home_team_id.astype(str) == tid]
        away = games[games.away_team_id.astype(str) == tid]
        gf = int(home.ft_home_goals.sum() + away.ft_away_goals.sum())
        ga = int(home.ft_away_goals.sum() + away.ft_home_goals.sum())
        wins = int((home.ft_home_goals > home.ft_away_goals).sum() + (away.ft_away_goals > away.ft_home_goals).sum())
        draws = int((home.ft_home_goals == home.ft_away_goals).sum() + (away.ft_home_goals == away.ft_away_goals).sum())
        played = len(home) + len(away)
        rows.append({'Team': name, 'P': played, 'W': wins, 'D': draws, 'L': played-wins-draws,
                     'GF': gf, 'GA': ga, 'GD': gf-ga, 'Pts': 3*wins+draws, '_team_id': tid})
    table = pd.DataFrame(rows).sort_values(['Pts','GD','GF','Team'], ascending=[False,False,False,True]).reset_index(drop=True)
    table.insert(0, 'Pos', range(1, len(table)+1))
    return table


def interval_history(matches, shots, fixture, team_ids, last_n=10):
    """Phase-safe, shot-derived interval totals for each team's latest league games.

    Uses only the featured season and games strictly before kickoff. Each team's
    history is selected independently; the featured fixture is always excluded.
    """
    prior = prior_matches(matches, fixture)
    match = matches.loc[matches.match_id.astype(str) == str(fixture['match_id'])]
    season = fixture.get('season_id') or (match.iloc[0]['season_id'] if not match.empty else None)
    if season is None:
        raise ValueError('Cannot identify featured match season for interval history')
    prior = prior.loc[prior.season_id.astype(str) == str(season)].sort_values(['date', 'match_id'])
    rows = []
    for team_id in team_ids:
        team_id = str(team_id)
        games = prior.loc[(prior.home_team_id.astype(str) == team_id) |
                          (prior.away_team_id.astype(str) == team_id)].tail(last_n)
        ids = set(games.match_id.astype(str))
        sample = shots.loc[shots.match_id.astype(str).isin(ids)].copy()
        if not sample.empty:
            sample['minute'] = pd.to_numeric(sample['minute'], errors='coerce')
            sample['xg'] = pd.to_numeric(sample['xg'], errors='coerce').fillna(0)
            sample['is_goal'] = sample['is_goal'].fillna(False).astype(bool)
        for start, end in INTERVALS:
            period = sample.loc[(sample.minute > start) & (sample.minute <= end)]
            own = period.loc[period.team_id.astype(str) == team_id]
            opposition = period.loc[period.team_id.astype(str) != team_id]
            gf = int(own.is_goal.sum())
            ga = int(opposition.is_goal.sum())
            xg = float(own.xg.sum())
            xga = float(opposition.xg.sum())
            n = len(games)
            rows.append({
                'team_id': team_id, 'Interval': f'{start}–{end}', 'Matches': n,
                'Shots for': len(own), 'Shots against': len(opposition),
                'Goals for': gf, 'Goals against': ga, 'Goal difference': gf-ga,
                'xG': round(xg, 3), 'xGA': round(xga, 3),
                'xG difference': round(xg-xga, 3),
                'Goals - xG': round(gf-xg, 3),
                'Goals against - xGA': round(ga-xga, 3),
                'xG per match': round(xg/n, 3) if n else None,
                'xGA per match': round(xga/n, 3) if n else None,
            })
    return pd.DataFrame(rows)
