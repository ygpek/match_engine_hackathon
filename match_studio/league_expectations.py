"""Phase-safe league standing vs synthetic competitive tier expectations."""
import pandas as pd

EXPECTED = {
    "title_contender": (1, 3),
    "european_contender": (4, 8),
    "mid_table": (9, 16),
    "relegation_battle": (17, 20),
}


def _standings(matches, teams):
    rows = []
    for team in teams.itertuples(index=False):
        tid = str(team.team_id)
        home = matches[matches.home_team_id.astype(str).eq(tid)]
        away = matches[matches.away_team_id.astype(str).eq(tid)]
        gf = int(home.ft_home_goals.sum() + away.ft_away_goals.sum())
        ga = int(home.ft_away_goals.sum() + away.ft_home_goals.sum())
        wins = int((home.ft_home_goals > home.ft_away_goals).sum() + (away.ft_away_goals > away.ft_home_goals).sum())
        draws = int((home.ft_home_goals == home.ft_away_goals).sum() + (away.ft_home_goals == away.ft_away_goals).sum())
        rows.append(dict(team_id=tid, team_name=str(team.team_name), played=len(home)+len(away),
                         points=3*wins+draws, goal_difference=gf-ga, goals_for=gf))
    table = pd.DataFrame(rows).sort_values(['points','goal_difference','goals_for','team_name'],
                                             ascending=[False,False,False,True]).reset_index(drop=True)
    table['position'] = range(1,len(table)+1)
    return table.set_index('team_id')


def build_league_expectations(all_matches, teams, team_profiles, metadata, post_snapshot, phase):
    if phase not in ('prematch','postmatch'):
        return None
    match_id = str(metadata['match_id'])
    selected = all_matches[all_matches.match_id.astype(str).eq(match_id)]
    if len(selected) != 1:
        raise ValueError('Expected exactly one featured match in historical_matches.parquet to resolve its season')
    featured = selected.iloc[0]
    season = str(featured.season_id)
    cutoff = pd.Timestamp(metadata['date'])
    eligible = all_matches.loc[
        all_matches.season_id.astype(str).eq(season)
        & pd.to_datetime(all_matches.date).lt(cutoff)
        & all_matches.match_id.astype(str).ne(match_id)
    ].copy()
    before = _standings(eligible, teams)
    after = None
    if phase == 'postmatch':
        # Use the final phase snapshot rather than unseen data from the source fixture.
        result = featured.copy()
        result['ft_home_goals'] = int(post_snapshot['home_goals'])
        result['ft_away_goals'] = int(post_snapshot['away_goals'])
        after = _standings(pd.concat([eligible, pd.DataFrame([result])],ignore_index=True), teams)
    categories = team_profiles.set_index(team_profiles.team_id.astype(str))['competitive_tier']
    output = {'season_id':season, 'basis':'Fixed synthetic competitive-tier position ranges, not statistical forecasts',
              'standings_tiebreakers':'Points, goal difference, goals for, team name', 'teams':{}}
    for side in ('home','away'):
        tid = str(metadata[f'{side}_team_id'])
        tier = str(categories.loc[tid])
        if tier not in EXPECTED:
            raise ValueError(f'Unexpected competitive tier: {tier}')
        lo,hi = EXPECTED[tier]
        def describe(table):
            r=table.loc[tid]
            pos=int(r['position'])
            return {'position':pos,'played':int(r['played']),'points':int(r['points']),
                    'goal_difference':int(r['goal_difference']),
                    'expectation_status':'above' if pos<lo else 'below' if pos>hi else 'within',
                    'positions_outside_range':lo-pos if pos<lo else pos-hi if pos>hi else 0}
        item={'team_id':tid,'team_name':str(before.loc[tid,'team_name']),
              'category':tier,'expected_position_min':lo,'expected_position_max':hi,
              'before_match':describe(before)}
        if after is not None:
            item['after_match']=describe(after)
            item['position_change']=item['before_match']['position']-item['after_match']['position']
            item['points_change']=item['after_match']['points']-item['before_match']['points']
        output['teams'][side]=item
    return output
