"""Evidence-only player storylines; no generative text or future-match leakage."""
from __future__ import annotations
import pandas as pd


def build_storylines(matches, player_stats, players, fixture, featured_stats=None, lineups=None,
                     phase='prematch', max_per_team=3):
    """Return up to max_per_team pre/post stories for each featured club.

    A player appearance is a positive-minutes player-match row. Rankings are
    season-specific; fixture is excluded until postmatch. 'Return' means an
    absence from recorded appearances, not a diagnosed injury.
    """
    cutoff = pd.Timestamp(fixture['date'])
    fixture_id = str(fixture['match_id'])
    all_matches = matches.copy()
    fixture_row = all_matches.loc[all_matches.match_id.astype(str).eq(fixture_id)]
    season = str(fixture_row.iloc[0]['season_id']) if not fixture_row.empty else None
    historical = all_matches.loc[(pd.to_datetime(all_matches.date) < cutoff) &
                                 all_matches.match_id.astype(str).ne(fixture_id)].copy()
    if season is not None:
        historical = historical.loc[historical.season_id.astype(str).eq(season)]
    hist_ids = set(historical.match_id.astype(str))
    stats = player_stats.loc[player_stats.match_id.astype(str).isin(hist_ids)].copy()
    stats = stats.loc[pd.to_numeric(stats.minutes_played,errors='coerce').fillna(0)>0]
    if not stats.empty:
        stats = stats.merge(historical[['match_id','date','home_team_id','away_team_id',
                                        'ft_home_goals','ft_away_goals']],on='match_id',how='left',validate='many_to_one')
    # Include the featured match only in the post-match evaluation.
    if phase == 'postmatch' and featured_stats is not None and not featured_stats.empty:
        current = featured_stats.loc[featured_stats.match_id.astype(str).eq(fixture_id)].copy()
        current = current.loc[pd.to_numeric(current.minutes_played,errors='coerce').fillna(0)>0]
        if not current.empty:
            current['date'] = cutoff
            current['home_team_id'] = fixture['home_team_id']
            current['away_team_id'] = fixture['away_team_id']
            # Read authoritative full-time score from historical fixture row.
            if not fixture_row.empty:
                current['ft_home_goals'] = int(fixture_row.iloc[0].ft_home_goals)
                current['ft_away_goals'] = int(fixture_row.iloc[0].ft_away_goals)
            else:
                current['ft_home_goals'] = pd.NA
                current['ft_away_goals'] = pd.NA
            stats = pd.concat([stats,current],ignore_index=True)
    if stats.empty:
        return {'phase':phase,'season_id':season,'stories':[], 'note':'No eligible player appearances.'}
    stats = stats.sort_values(['date','match_id']).drop_duplicates(['match_id','player_id'],keep='last')
    player_lookup = players.drop_duplicates('player_id').set_index('player_id')
    totals = stats.groupby('player_id').agg(goals=('goals','sum'),assists=('assists','sum'),
                                           appearances=('match_id','nunique'))
    scorers = totals.sort_values(['goals','assists'],ascending=False)
    league_top = int(scorers.goals.max())
    leaders = scorers.loc[scorers.goals.eq(league_top)]
    candidates = []
    for team_id in (str(fixture['home_team_id']),str(fixture['away_team_id'])):
        squad = players.loc[players.team_id.astype(str).eq(team_id)]
        for player in squad.itertuples(index=False):
            pid = player.player_id
            if pid not in totals.index: continue
            name = getattr(player,'player_name',None) or getattr(player,'name',None) or str(pid)
            pos = str(getattr(player,'primary_position',''))
            rows = stats.loc[stats.player_id.eq(pid)].sort_values(['date','match_id'])
            if rows.empty: continue
            g = int(totals.loc[pid,'goals']); a = int(totals.loc[pid,'assists'])
            base = {'player_id':str(pid),'player_name':str(name),'team_id':team_id,
                    'position':pos,'phase':phase}
            def add(kind,priority,evidence):
                candidates.append({**base,'type':kind,'priority':priority,'evidence':evidence})
            if g>=3 and g>=league_top-2 and pos!='GK':
                rank = 1 + int((scorers.goals>g).sum())
                add('golden_boot_race', 90 if rank==1 else 80,
                    {'season_goals':g,'league_rank':rank,'leader_goals':league_top,
                     'goals_behind_leader':league_top-g,'tied_leaders':len(leaders)})
            if pos in ('ST','AMW'):
                last = rows.tail(8)
                streak=0
                for x in reversed(rows.goals.tolist()):
                    if x>0: streak+=1
                    else: break
                if streak>=3:
                    add('scoring_streak',min(85,50+streak*6),
                        {'consecutive_scoring_appearances':streak,
                         'goals_in_streak':int(rows.tail(streak).goals.sum())})
                drought=0
                for x in reversed(rows.goals.tolist()):
                    if x==0:drought+=1
                    else:break
                if drought>=5 and g>=2:
                    add('goal_drought',min(65,35+drought*4),
                        {'consecutive_goalless_appearances':drought,
                         'season_goals':g,'last_8_shots':int(last.shots.sum())})
            if pos=='GK':
                recent=rows.tail(5).copy()
                if recent.ft_home_goals.notna().all() and recent.ft_away_goals.notna().all():
                    conceded = recent.apply(lambda r: r.ft_away_goals if str(r.team_id)==str(r.home_team_id)
                                            else r.ft_home_goals,axis=1)
                    eligible = pd.to_numeric(recent.minutes_played,errors='coerce').ge(60)
                    clean = conceded.eq(0) & eligible
                    count=int(clean.sum())
                    streak=0
                    for val in reversed(clean.tolist()):
                        if val:streak+=1
                        else:break
                    if count>=3 or streak>=3:
                        add('goalkeeper_clean_sheets',60+min(20,count*4),
                            {'clean_sheets_last_5_appearances':count,
                             'appearances_considered':len(recent),
                             'consecutive_clean_sheets':streak,
                             'minimum_minutes_for_credit':60})
            if a>=3 and pos in ('CM','AMW','WB','ST'):
                add('creative_playmaker',min(70,35+a*4),
                    {'season_assists':a,'last_5_assists':int(rows.tail(5).assists.sum())})
            # Missed team fixtures are not proof of injury; no medical inference.
            team_fixtures = historical.loc[(historical.home_team_id.astype(str)==team_id) |
                                           (historical.away_team_id.astype(str)==team_id)]
            if phase=='prematch' and lineups is not None and not lineups.empty:
                starting = lineups.loc[lineups.player_id.astype(str).eq(str(pid)) &
                                       lineups.started.astype(bool)]
                if not starting.empty and len(rows):
                    last_played = pd.Timestamp(rows.iloc[-1].date)
                    missed = int(((pd.to_datetime(team_fixtures.date)>last_played) &
                                  (pd.to_datetime(team_fixtures.date)<cutoff)).sum())
                    if missed>=3:
                        add('return_from_absence',min(75,45+missed*4),
                            {'consecutive_team_matches_missed':missed,
                             'confirmed_starting':True,'absence_reason':'unknown'})
            if phase=='postmatch' and not rows.empty and str(rows.iloc[-1].match_id)==fixture_id:
                previous=rows.iloc[:-1]
                if not previous.empty:
                    last_played=pd.Timestamp(previous.iloc[-1].date)
                    missed=int(((pd.to_datetime(team_fixtures.date)>last_played)&
                                (pd.to_datetime(team_fixtures.date)<cutoff)).sum())
                    if missed>=3:
                        add('return_from_absence',min(75,45+missed*4),
                            {'consecutive_team_matches_missed':missed,
                             'minutes_on_return':int(rows.iloc[-1].minutes_played),
                             'absence_reason':'unknown'})
    # Expose participation status explicitly, and avoid stories about unused
    # substitutes in postmatch analysis. Prematch bench players may be discussed
    # only as bench options, never as members of the starting XI.
    lineup_status = {}
    if lineups is not None and not lineups.empty:
        for row in lineups.itertuples(index=False):
            pid = str(row.player_id)
            if bool(row.started):
                status = "starter"
            elif phase == "postmatch" and int(row.minute_on) < 90:
                status = "substitute_used"
            else:
                status = "unused_substitute"
            lineup_status[pid] = status
    for candidate in candidates:
        candidate["lineup_status"] = lineup_status.get(candidate["player_id"], "not_in_matchday_lineup")
        candidate["featured_match_participated"] = (
            phase == "postmatch" and candidate["lineup_status"] in ("starter", "substitute_used")
        )
    if phase == "postmatch":
        candidates = [c for c in candidates if c["featured_match_participated"]]
    selected=[]
    for team in (str(fixture['home_team_id']),str(fixture['away_team_id'])):
        team_stories=sorted((c for c in candidates if c['team_id']==team),
                            key=lambda x:(x["lineup_status"] != "starter", -x["priority"], x["player_id"], x["type"]))
        used=set()
        for story in team_stories:
            if story['player_id'] in used:continue
            related = [{'type':other['type'],'evidence':other['evidence']}
                       for other in team_stories
                       if other['player_id']==story['player_id'] and other is not story]
            selected.append({**story,'related_storylines':related});used.add(story['player_id'])
            if len(used)>=max_per_team:break
    return {'phase':phase,'season_id':season,'historical_cutoff_exclusive':str(cutoff.date()),
            'stories':selected,'candidate_count':len(candidates),
            'note':'Only recorded appearances; absences are not identified as injuries. '
                   'GK clean sheets use full-match team goals conceded and at least 60 minutes.'}
