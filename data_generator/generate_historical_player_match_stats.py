"""Step 5: historical player appearances and match statistics.

Run after generate_players.py, generate_historical_matches.py and
      generate_historical_team_stats.py.

Inputs in ./data: players, player_profiles, historical_matches,
                  historical_shots, historical_team_stats (.parquet).
Outputs: historical_lineups.parquet, historical_player_match_stats.parquet.

The shot table is authoritative. Team counts are allocated to eligible players
and reconcile exactly, while physical figures are synthetic estimates.
Requires numpy, pandas, pyarrow.
"""
from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / 'data'
SEED = 46
# One GK, two CB, two WB, three CM, two AMW, one ST = 11.
STARTING_XI = {'GK': 1, 'CB': 2, 'WB': 2, 'CM': 3, 'AMW': 2, 'ST': 1}
MAX_SUBS = 5
SUB_MINUTE_RANGE = (52, 84)
# Per-90 synthetic tracking baselines, by position.
DISTANCE_KM_P90 = {'GK': 4.5, 'CB': 9.2, 'WB': 10.6, 'CM': 11.2, 'AMW': 10.2, 'ST': 9.6}
SPRINTS_P90 = {'GK': 2, 'CB': 13, 'WB': 30, 'CM': 20, 'AMW': 34, 'ST': 28}
# Player metric -> team metric and skill dimension, allocation positional weight.
ALLOCATIONS = {
    'passes_attempted': ('passing', {'GK': .35, 'CB': 1.0, 'WB': 1.05, 'CM': 1.6, 'AMW': .95, 'ST': .50}),
    'tackles': ('defensive', {'GK': .01, 'CB': 1.15, 'WB': 1.2, 'CM': 1.2, 'AMW': .5, 'ST': .3}),
    'interceptions': ('defensive', {'GK': .02, 'CB': 1.4, 'WB': 1.0, 'CM': 1.25, 'AMW': .4, 'ST': .2}),
    'recoveries': ('defensive', {'GK': .2, 'CB': 1.1, 'WB': 1.0, 'CM': 1.4, 'AMW': .75, 'ST': .45}),
    'clearances': ('defensive', {'GK': .3, 'CB': 2.8, 'WB': 1.5, 'CM': .45, 'AMW': .15, 'ST': .12}),
    'fouls': ('defensive', {'GK': .03, 'CB': 1.1, 'WB': 1.0, 'CM': 1.2, 'AMW': .7, 'ST': .6}),
    'progressive_passes': ('passing', {'GK': .15, 'CB': .8, 'WB': 1.0, 'CM': 1.7, 'AMW': 1.1, 'ST': .3}),
    'passes_final_third': ('passing', {'GK': .02, 'CB': .3, 'WB': 1.0, 'CM': 1.8, 'AMW': 1.6, 'ST': .5}),
}


def weighted_choice(rng, eligible, players_by_id, profiles_by_id, skill, position_weights):
    weights = []
    for pid in eligible:
        pos = players_by_id[pid]['primary_position']
        ability = float(profiles_by_id[pid][f'{skill}_ability'])
        weights.append(max(.001, position_weights.get(pos, .1) * np.exp((ability - 60) / 35)))
    weights = np.asarray(weights, dtype=float)
    return rng.choice(eligible, p=weights / weights.sum())


def build_lineup(rng, match_id, team_id, squad, profiles_by_id):
    """Choose XI by role, then substitutes; return player appearance windows."""
    starters, bench = [], []
    for pos, count in STARTING_XI.items():
        candidates = [p for p in squad if p['primary_position'] == pos]
        if len(candidates) < count:
            raise ValueError(f'{team_id} lacks players for {pos}')
        quality = np.array([profiles_by_id[p['player_id']]['overall_ability'] for p in candidates], dtype=float)
        quality += rng.normal(0, 9, len(quality))  # Rotation / match-to-match selection.
        order = np.argsort(-quality)
        starters.extend(candidates[int(i)]['player_id'] for i in order[:count])
        bench.extend(candidates[int(i)]['player_id'] for i in order[count:])
    windows = {pid: [0, 90] for pid in starters}
    # Not every match needs five substitutions.
    n_subs = int(rng.integers(2, MAX_SUBS + 1))
    off_pool = [pid for pid in starters if next(p for p in squad if p['player_id'] == pid)['primary_position'] != 'GK']
    rng.shuffle(off_pool)
    used_on = set()
    substitutions = []
    for off in off_pool[:n_subs]:
        off_pos = next(p for p in squad if p['player_id'] == off)['primary_position']
        options = [pid for pid in bench if pid not in used_on and
                   next(p for p in squad if p['player_id'] == pid)['primary_position'] == off_pos]
        if not options:
            continue
        on = str(rng.choice(options))
        minute = int(rng.integers(*SUB_MINUTE_RANGE))
        windows[off][1] = minute
        windows[on] = [minute, 90]
        used_on.add(on)
        substitutions.append((off, on, minute))
    return windows, starters, substitutions


def generate(players, profiles, matches, shots, team_stats, seed=SEED):
    rng = np.random.default_rng(seed)
    player_map = players.set_index('player_id').to_dict('index')
    ability_map = profiles.set_index('player_id').to_dict('index')
    squads = {tid: grp.to_dict('records') for tid, grp in players.groupby('team_id')}
    shots_by_match_team = {(mid, tid): grp.sort_values(['minute', 'second', 'shot_id']).to_dict('records')
                           for (mid, tid), grp in shots.groupby(['match_id', 'team_id'])}
    stats_index = team_stats.set_index(['match_id', 'team_id', 'period'])
    lineup_rows, stat_rows, discipline_rows = [], [], []
    shot_attributions = {}
    team_fields = ['passes_attempted', 'passes_completed', 'progressive_passes',
                   'passes_final_third', 'tackles', 'interceptions', 'recoveries',
                   'clearances', 'fouls', 'yellow_cards', 'red_cards']
    for m in matches.itertuples(index=False):
        for team in (m.home_team_id, m.away_team_id):
            windows, starters, subs = build_lineup(rng, m.match_id, team, squads[team], ability_map)
            # Schedule dismissals before allocating shots or player actions.
            # A red card changes the appearance window, so later shots and
            # second-half statistics cannot be attributed to that player.
            scheduled_reds = []
            for half in (1, 2):
                team_half = stats_index.loc[(m.match_id, team, half)]
                for _ in range(int(team_half.red_cards)):
                    low, high = (12, 44) if half == 1 else (55, 87)
                    candidates = [pid for pid in starters if windows[pid][1] == 90 and
                                  player_map[pid]['primary_position'] != 'GK' and
                                  pid not in [x[0] for x in scheduled_reds]]
                    if not candidates:
                        raise ValueError(f'Cannot schedule a consistent dismissal for {m.match_id} {team}')
                    weights = np.array([{'CB': 2.0, 'WB': 1.6, 'CM': 1.4, 'AMW': .6, 'ST': .7}.get(
                        player_map[pid]['primary_position'], 1.) for pid in candidates], dtype=float)
                    pid = str(rng.choice(candidates, p=weights / weights.sum()))
                    minute = int(rng.integers(low, high + 1))
                    windows[pid][1] = minute
                    scheduled_reds.append((pid, minute, half))
                    discipline_rows.append(dict(match_id=m.match_id, team_id=team, player_id=pid,
                        minute=minute, second=59, period=half, event_type='red_card',
                        description='Direct red card'))
            # Only players with actual minutes have a performance row.
            records = {}
            for pid, (start, end) in windows.items():
                p = player_map[pid]
                records[pid] = dict(match_id=m.match_id, season_id=m.season_id,
                    team_id=team, player_id=pid, position_played=p['primary_position'],
                    started=pid in starters, minute_on=start, minute_off=end,
                    minutes_played=end - start,
                    **{field: 0 for field in team_fields},
                    shots=0, shots_on_target=0, goals=0, xg=0., assists=0, xa=0.,
                    key_passes=0, progressive_carries=0, successful_dribbles=0,
                    dispossessions=0, distance_km=0., sprints=0, accelerations=0)
                lineup_rows.append(dict(match_id=m.match_id, team_id=team,
                    player_id=pid, position=p['primary_position'],
                    started=pid in starters, minute_on=start, minute_off=end))

            def active(minute):
                return [pid for pid, (start, end) in windows.items() if start <= minute < end]

            # Allocate actual shots to players on the pitch at their timestamp.
            for shot in shots_by_match_team.get((m.match_id, team), []):
                eligible = active(int(shot['minute']) - 1 + int(shot['second']) / 60)
                if not eligible:
                    raise ValueError(f'No active shooter for {shot["shot_id"]}')
                shooter = weighted_choice(rng, eligible, player_map, ability_map, 'attacking',
                    {'GK': .001, 'CB': .12, 'WB': .35, 'CM': .55, 'AMW': 1.65, 'ST': 2.7})
                shot_attributions[str(shot['shot_id'])] = {'player_id': str(shooter), 'assist_player_id': None}
                r = records[shooter]
                r['shots'] += 1
                r['shots_on_target'] += int(shot['is_on_target'])
                r['goals'] += int(shot['is_goal'])
                r['xg'] += float(shot['xg'])
                # Chance creation: not every shot is assisted; xA credits a creator.
                others = [pid for pid in eligible if pid != shooter]
                if others and rng.random() < .67:
                    creator = weighted_choice(rng, others, player_map, ability_map, 'passing',
                        {'GK': .02, 'CB': .2, 'WB': .7, 'CM': 1.4, 'AMW': 1.9, 'ST': .65})
                    shot_attributions[str(shot['shot_id'])]['assist_player_id'] = str(creator) if shot['is_goal'] else None
                    records[creator]['key_passes'] += 1
                    records[creator]['xa'] += float(shot['xg'])
                    if shot['is_goal']:
                        records[creator]['assists'] += 1

            # Allocate additive half totals, constrained by each player's time on pitch.
            for half in (1, 2):
                row = stats_index.loc[(m.match_id, team, half)]
                a, b = (0, 45) if half == 1 else (45, 90)
                eligible = [pid for pid, (start, end) in windows.items() if min(end, b) > max(start, a)]
                exposure = {pid: min(windows[pid][1], b) - max(windows[pid][0], a) for pid in eligible}
                def allocate(total, skill, weights, pool=None):
                    pool = pool or eligible
                    if not pool:
                        if total:
                            raise ValueError('No players to allocate nonzero statistic')
                        return {}
                    probs = np.array([exposure[pid] * weights.get(player_map[pid]['primary_position'], .1)
                                      * np.exp((ability_map[pid][f'{skill}_ability'] - 60) / 35)
                                      for pid in pool], dtype=float)
                    probs /= probs.sum()
                    return dict(zip(pool, rng.multinomial(int(total), probs)))

                attempts = allocate(row.passes_attempted, *ALLOCATIONS['passes_attempted'])
                # Completed passes are sampled within each player's attempted passes,
                # and conditioned on the team completed total.
                tickets = []
                for pid, n in attempts.items():
                    tickets.extend([pid] * int(n))
                rng.shuffle(tickets)
                completed = {pid: 0 for pid in eligible}
                # Completion ability influences selection without breaking team totals.
                if tickets:
                    ticket_weights = np.array([np.exp((ability_map[pid]['passing_ability'] - 60) / 32)
                                               for pid in tickets])
                    chosen = rng.choice(len(tickets), size=int(row.passes_completed),
                                        replace=False, p=ticket_weights / ticket_weights.sum())
                    for ix in chosen:
                        completed[tickets[int(ix)]] += 1
                for pid in eligible:
                    records[pid]['passes_attempted'] += attempts.get(pid, 0)
                    records[pid]['passes_completed'] += completed[pid]

                # Progressive passes must be a subset of completed passes.
                prog_tickets = []
                for pid, n in completed.items():
                    prog_tickets.extend([pid] * int(n))
                rng.shuffle(prog_tickets)
                if int(row.progressive_passes) > len(prog_tickets):
                    raise ValueError('Team progressive passes exceed completed passes')
                prog_weights = np.array([np.exp((ability_map[pid]['passing_ability'] - 60) / 32)
                                         * ALLOCATIONS['progressive_passes'][1][player_map[pid]['primary_position']]
                                         for pid in prog_tickets])
                prog_idx = rng.choice(len(prog_tickets), size=int(row.progressive_passes),
                                      replace=False, p=prog_weights / prog_weights.sum()) if len(prog_tickets) else []
                progressive = {pid: 0 for pid in eligible}
                for ix in prog_idx:
                    progressive[prog_tickets[int(ix)]] += 1
                for pid, n in progressive.items():
                    records[pid]['progressive_passes'] += n
                # Final-third passes are a subset of progressive passes in this
                # simplified team generator (a modeling convention, not a football rule).
                final_tickets = [pid for pid, n in progressive.items() for _ in range(n)]
                rng.shuffle(final_tickets)
                for pid in final_tickets[:int(row.passes_final_third)]:
                    records[pid]['passes_final_third'] += 1

                for metric in ('tackles', 'interceptions', 'recoveries', 'clearances', 'fouls'):
                    allocated = allocate(row[metric], *ALLOCATIONS[metric])
                    for pid, count in allocated.items():
                        records[pid][metric] += count
                # Yellow cards are individual, time-stamped and constrained
                # to actual on-pitch windows. No player receives two yellows
                # without a dismissal; second-yellow logic is a future extension.
                card_pool = [pid for pid in eligible if records[pid]['yellow_cards'] == 0]
                for _ in range(int(row.yellow_cards)):
                    if not card_pool:
                        raise ValueError(f'Too many yellows for {m.match_id} {team}')
                    weights = np.array([1 + records[pid]['fouls'] for pid in card_pool], dtype=float)
                    pid = str(rng.choice(card_pool, p=weights / weights.sum()))
                    card_pool.remove(pid)
                    records[pid]['yellow_cards'] += 1
                    first = max(a + 1, windows[pid][0] + 1)
                    last = min(b, windows[pid][1])
                    minute = int(rng.integers(first, last + 1))
                    discipline_rows.append(dict(match_id=m.match_id, team_id=team,
                        player_id=pid, minute=minute, second=10, period=half,
                        event_type='yellow_card', description='Yellow card'))
            for pid, minute, half in scheduled_reds:
                records[pid]['red_cards'] += 1

            for pid, r in records.items():
                pos = player_map[pid]['primary_position']
                mins = r['minutes_played']
                factor = mins / 90
                physical = ability_map[pid]['physical_ability']
                r['distance_km'] = round(max(0, rng.normal(
                    DISTANCE_KM_P90[pos] * factor * (1 + .004 * (physical - 60)),
                    .45 * np.sqrt(factor))), 3)
                r['sprints'] = int(rng.poisson(max(0, SPRINTS_P90[pos] * factor * (1 + .008 * (physical - 60)))))
                r['accelerations'] = int(rng.poisson(max(0, r['sprints'] * 1.7)))
                # Possession actions not currently included in team totals are
                # generated independently; they are not team-reconciled.
                r['progressive_carries'] = int(rng.poisson(max(0, factor *
                    {'GK': .05, 'CB': .5, 'WB': 2.2, 'CM': 1.9, 'AMW': 3.3, 'ST': 1.1}[pos]
                    * np.exp((ability_map[pid]['possession_ability'] - 60) / 45))))
                r['successful_dribbles'] = int(rng.poisson(max(0, factor *
                    {'GK': .01, 'CB': .15, 'WB': .8, 'CM': .65, 'AMW': 2.0, 'ST': .8}[pos]
                    * np.exp((ability_map[pid]['possession_ability'] - 60) / 45))))
                r['dispossessions'] = int(rng.poisson(max(0, factor *
                    {'GK': .02, 'CB': .2, 'WB': .55, 'CM': .7, 'AMW': 1.25, 'ST': 1.0}[pos])))
                stat_rows.append(r)
    attribution = pd.DataFrame([{'shot_id': k, **v} for k, v in shot_attributions.items()])
    return pd.DataFrame(lineup_rows), pd.DataFrame(stat_rows), attribution, pd.DataFrame(
        discipline_rows, columns=['match_id','team_id','player_id','minute','second','period','event_type','description'])


def validate(lineups, player_stats, matches, shots, team_stats):
    assert not lineups.duplicated(['match_id', 'player_id']).any()
    assert not player_stats.duplicated(['match_id', 'player_id']).any()
    assert (player_stats.minutes_played > 0).all()
    assert (player_stats.passes_completed <= player_stats.passes_attempted).all()
    assert (player_stats.progressive_passes <= player_stats.passes_completed).all()
    assert (player_stats.passes_final_third <= player_stats.progressive_passes).all()
    assert (player_stats.goals <= player_stats.shots_on_target).all()
    assert (player_stats.shots_on_target <= player_stats.shots).all()
    assert (player_stats.assists <= player_stats.key_passes).all()
    assert (player_stats.xg >= 0).all()
    assert (player_stats.xa >= 0).all()
    assert (lineups.groupby(['match_id', 'team_id']).started.sum() == 11).all()
    assert (player_stats.groupby(['match_id', 'team_id']).minutes_played.sum() <= 990).all()
    assert (lineups.groupby(['match_id', 'team_id']).size() <= 16).all()
    agg = player_stats.groupby(['match_id', 'team_id'])
    shot_agg = shots.groupby(['match_id', 'team_id']).agg(shots=('shot_id', 'size'),
        shots_on_target=('is_on_target', 'sum'), goals=('is_goal', 'sum'), xg=('xg', 'sum'))
    for metric in ('shots', 'shots_on_target', 'goals'):
        assert agg[metric].sum().reindex(shot_agg.index, fill_value=0).equals(shot_agg[metric].astype('int64')), metric
    assert np.allclose(agg.xg.sum().reindex(shot_agg.index, fill_value=0), shot_agg.xg)
    team_total = team_stats.groupby(['match_id', 'team_id']).sum(numeric_only=True)
    for metric in ('passes_attempted', 'passes_completed', 'progressive_passes',
                   'passes_final_third', 'tackles', 'interceptions', 'recoveries',
                   'clearances', 'fouls', 'yellow_cards', 'red_cards'):
        actual = agg[metric].sum().reindex(team_total.index, fill_value=0)
        assert np.array_equal(actual.to_numpy(), team_total[metric].to_numpy()), metric


def main():
    inputs = {name: pd.read_parquet(DATA_DIR / f'{name}.parquet') for name in
        ('players', 'player_profiles', 'historical_matches', 'historical_shots', 'historical_team_stats')}
    lineups, stats, attribution, discipline = generate(inputs['players'], inputs['player_profiles'],
        inputs['historical_matches'], inputs['historical_shots'], inputs['historical_team_stats'])
    validate(lineups, stats, inputs['historical_matches'], inputs['historical_shots'], inputs['historical_team_stats'])
    enriched_shots = inputs['historical_shots'].drop(columns=['player_id', 'assist_player_id'], errors='ignore').merge(
        attribution, on='shot_id', how='left', validate='one_to_one')
    if enriched_shots.player_id.isna().any():
        raise ValueError('Some shots lack player attribution')
    enriched_shots.to_parquet(DATA_DIR / 'historical_shots.parquet', index=False)
    lineups.to_parquet(DATA_DIR / 'historical_lineups.parquet', index=False)
    discipline.to_parquet(DATA_DIR / 'historical_disciplinary_events.parquet', index=False)
    stats.to_parquet(DATA_DIR / 'historical_player_match_stats.parquet', index=False)
    print(f'Generated {len(lineups):,} appearances and {len(stats):,} player-match records')
    print('Average appearances per match:', round(len(stats) / len(inputs['historical_matches']), 2))
    print('Validation passed; output saved to', DATA_DIR)
    print(stats.head(5).to_string(index=False))

if __name__ == '__main__':
    main()
