"""Generate a detailed, internally consistent featured match from a historical replay.

This MVP replays one existing simulated historical match rather than generating
an independent future fixture. It preserves the underlying shots, goals, xG,
lineups and substitutions, and creates event/snapshot views for Match Studio.

Usage:
 python featured_match/generate_featured_match.py
 python featured_match/generate_featured_match.py --match-id <historical_match_id>
 python featured_match/generate_featured_match.py --live-minute 67

Reads ../data_generator/data; writes ./data.
"""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "data_generator" / "data"
DEST = HERE / "data"


def clean(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    return value


def load():
    files = [
        "teams",
        "historical_matches",
        "historical_shots",
        "historical_lineups",
        "historical_team_stats",
        "players",
    ]
    result = {}
    for name in files:
        path = SOURCE / f"{name}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}. Run data_generator/generate_data.py first.")
        result[name] = pd.read_parquet(path)
    return result


def select_match(matches, shots, explicit_id, mode, rng, previous_id=None):
    """Choose a complete historical replay; never randomize the match's facts."""
    if explicit_id is not None:
        found = matches[matches.match_id.astype(str) == str(explicit_id)]
        if len(found) != 1:
            raise ValueError(f"Expected exactly one match with ID {explicit_id!r}; found {len(found)}")
        return found.iloc[0]

    eligible = matches.copy()
    if previous_id is not None and len(eligible) > 1:
        remaining = eligible[eligible.match_id.astype(str) != str(previous_id)]
        if not remaining.empty:
            eligible = remaining

    if mode == "dramatic":
        # Original dramatic pattern: home leads 1-0 at HT, away equalizes
        # in minutes 46-75, home scores the winner in minutes 76-90.
        goals = shots[shots.is_goal.astype(bool)]
        candidates = []
        for m in eligible.itertuples(index=False):
            if (m.ht_home_goals, m.ht_away_goals, m.ft_home_goals, m.ft_away_goals) != (1, 0, 2, 1):
                continue
            match_goals = goals[goals.match_id.astype(str) == str(m.match_id)]
            equalizer = match_goals[(match_goals.team_id == m.away_team_id) & match_goals.minute.between(46, 75)]
            winner = match_goals[(match_goals.team_id == m.home_team_id) & (match_goals.minute >= 76)]
            if not equalizer.empty and not winner.empty:
                candidates.append(m.match_id)
        if candidates:
            eligible = eligible[eligible.match_id.isin(candidates)]
        else:
            print("No qualifying dramatic match found; selecting randomly from available fixtures.")

    selected_index = int(rng.integers(0, len(eligible)))
    return eligible.iloc[selected_index]


def build_events(m, shots, lineups, team_stats, rng):
    events = []
    for s in shots.sort_values(["minute", "second", "shot_id"]).itertuples(index=False):
        events.append(
            dict(
                event_id=f"SHOT_{s.shot_id}",
                match_id=m.match_id,
                minute=int(s.minute),
                second=int(s.second),
                team_id=s.team_id,
                event_type="goal" if s.is_goal else "shot",
                shot_id=s.shot_id,
                player_id=None,
                xg=float(s.xg),
                is_on_target=bool(s.is_on_target),
                is_goal=bool(s.is_goal),
                description="Goal" if s.is_goal else ("Shot on target" if s.is_on_target else "Shot off target"),
            )
        )
    for row in lineups.itertuples(index=False):
        if not bool(row.started) and int(row.minute_on) < 90:
            events.append(
                dict(
                    event_id=f"SUB_{row.team_id}_{row.player_id}",
                    match_id=m.match_id,
                    minute=int(row.minute_on),
                    second=0,
                    team_id=row.team_id,
                    event_type="substitution_on",
                    shot_id=None,
                    player_id=row.player_id,
                    xg=None,
                    is_on_target=None,
                    is_goal=False,
                    description="Player substituted on",
                )
            )
        if bool(row.started) and int(row.minute_off) < 90:
            events.append(
                dict(
                    event_id=f"OFF_{row.team_id}_{row.player_id}",
                    match_id=m.match_id,
                    minute=int(row.minute_off),
                    second=0,
                    team_id=row.team_id,
                    event_type="substitution_off",
                    shot_id=None,
                    player_id=row.player_id,
                    xg=None,
                    is_on_target=None,
                    is_goal=False,
                    description="Player substituted off",
                )
            )
    # Other team statistics are half totals, so do NOT fabricate exact timestamps
    # for passes/tackles/cards. Store them as period aggregates instead.
    events.extend(
        [
            dict(
                event_id="KICKOFF",
                match_id=m.match_id,
                minute=0,
                second=0,
                team_id=None,
                event_type="kickoff",
                shot_id=None,
                player_id=None,
                xg=None,
                is_on_target=None,
                is_goal=False,
                description="Kickoff",
            ),
            dict(
                event_id="HALFTIME",
                match_id=m.match_id,
                minute=45,
                second=59,
                team_id=None,
                event_type="halftime",
                shot_id=None,
                player_id=None,
                xg=None,
                is_on_target=None,
                is_goal=False,
                description="Halftime",
            ),
            dict(
                event_id="FULLTIME",
                match_id=m.match_id,
                minute=90,
                second=59,
                team_id=None,
                event_type="fulltime",
                shot_id=None,
                player_id=None,
                xg=None,
                is_on_target=None,
                is_goal=False,
                description="Full time",
            ),
        ]
    )
    events = pd.DataFrame(events).sort_values(["minute", "second", "event_id"]).reset_index(drop=True)
    events.insert(0, "sequence", np.arange(1, len(events) + 1))
    return events


def snapshot(m, shots, lineups, team_stats, minute, phase):
    # Snapshot at end of minute N, never include later events.
    seen = shots[shots.minute <= minute]
    scores = {t: int(seen[(seen.team_id == t) & seen.is_goal].shape[0]) for t in (m.home_team_id, m.away_team_id)}
    shot_summary = {}
    for t in (m.home_team_id, m.away_team_id):
        subset = seen[seen.team_id == t]
        shot_summary[t] = dict(
            shots=len(subset),
            shots_on_target=int(subset.is_on_target.sum()),
            xg=round(float(subset.xg.sum()), 3),
            goals=scores[t],
        )
    if minute == 0:
        # No historical outcomes should be exposed before kickoff.
        pass
    active = lineups[
        (lineups.minute_on <= minute) & ((lineups.minute_off > minute) | ((minute == 90) & (lineups.minute_off == 90)))
    ]
    if minute == 0:
        active = lineups[lineups.started]
    # Only completed halves may be used as team aggregates.
    completed_halves = 0 if minute < 45 else (1 if minute < 90 else 2)
    available_stats = team_stats[team_stats.period <= completed_halves]
    half_aggregates = {}
    for team in (m.home_team_id, m.away_team_id):
        sub = available_stats[available_stats.team_id == team]
        half_aggregates[team] = {
            col: clean(sub[col].sum())
            for col in ("passes_attempted", "passes_completed", "tackles", "interceptions", "recoveries", "corners")
        }
        half_aggregates[team]["complete_through_minute"] = completed_halves * 45
    return dict(
        phase=phase,
        minute=minute,
        home_goals=scores[m.home_team_id],
        away_goals=scores[m.away_team_id],
        shots=shot_summary,
        active_player_ids={
            str(t): active[active.team_id == t].player_id.tolist() for t in (m.home_team_id, m.away_team_id)
        },
        completed_half_stats=half_aggregates,
        note="Pass/defensive totals shown only for completed halves; shot counts update live.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--match-id", default=None, help="Replay a specific historical match ID")
    parser.add_argument("--mode", choices=("random", "dramatic"), default="random",
                        help="Selection mode when --match-id is not supplied")
    parser.add_argument("--seed", type=int, default=None,
                        help="Optional reproducible random seed; omit for fresh randomness")
    parser.add_argument("--allow-repeat", action="store_true",
                        help="Allow selection of the previously generated featured match")
    parser.add_argument("--live-minute", type=int, default=67)
    args = parser.parse_args()
    if not 46 <= args.live_minute <= 89:
        parser.error("--live-minute must be between 46 and 89")
    data = load()
    rng = np.random.default_rng(args.seed)
    previous_id = None
    previous_path = DEST / "featured_match.json"
    if not args.allow_repeat and args.match_id is None and previous_path.exists():
        try:
            previous_id = json.loads(previous_path.read_text(encoding="utf-8")).get("match_id")
        except (ValueError, OSError):
            pass
    m = select_match(data["historical_matches"], data["historical_shots"],
                     args.match_id, args.mode, rng, previous_id)
    mid = m.match_id
    shots = data["historical_shots"].query("match_id == @mid").copy()
    lineups = data["historical_lineups"].query("match_id == @mid").copy()
    team_stats = data["historical_team_stats"].query("match_id == @mid").copy()
    if len(lineups[lineups.started]) != 22:
        raise ValueError("Featured match must have 22 starters")
    assert int(shots.is_goal.sum()) == int(m.ft_home_goals + m.ft_away_goals)
    events = build_events(m, shots, lineups, team_stats, rng)
    checkpoints = [("prematch", 0), ("halftime", 45), ("live", args.live_minute), ("postmatch", 90)]
    snapshots = [snapshot(m, shots, lineups, team_stats, minute, phase) for phase, minute in checkpoints]
    assert snapshots[1]["home_goals"] == int(m.ht_home_goals)
    assert snapshots[1]["away_goals"] == int(m.ht_away_goals)
    assert snapshots[-1]["home_goals"] == int(m.ft_home_goals)
    assert snapshots[-1]["away_goals"] == int(m.ft_away_goals)
    assert len(events[events.event_type.isin(["goal", "shot"])]) == len(shots)
    DEST.mkdir(parents=True, exist_ok=True)
    events.to_parquet(DEST / "featured_match_events.parquet", index=False)
    lineups.to_parquet(DEST / "featured_match_lineups.parquet", index=False)
    team_stats.to_parquet(DEST / "featured_match_team_half_stats.parquet", index=False)
    shots.to_parquet(DEST / "featured_match_shots.parquet", index=False)
    metadata = dict(
        match_id=str(mid),
        source="historical_replay",
        selection_mode="explicit" if args.match_id is not None else args.mode,
        selection_seed=args.seed,
        date=clean(pd.Timestamp(m.date)),
        home_team_id=m.home_team_id,
        away_team_id=m.away_team_id,
        live_minute=args.live_minute,
        ht_score=[int(m.ht_home_goals), int(m.ht_away_goals)],
        ft_score=[int(m.ft_home_goals), int(m.ft_away_goals)],
    )
    (DEST / "featured_match.json").write_text(json.dumps(clean(metadata), indent=2), encoding="utf-8")
    (DEST / "featured_match_snapshots.json").write_text(json.dumps(clean(snapshots), indent=2), encoding="utf-8")
    print(f"Featured historical replay: {mid} | {m.home_team_id} vs {m.away_team_id}")
    print(f"HT {m.ht_home_goals}-{m.ht_away_goals}; FT {m.ft_home_goals}-{m.ft_away_goals}")
    print(f"{len(events)} events, {len(lineups)} lineup entries -> {DEST}")
    print("NOTE: This is a replay of a historical match, not an independent future fixture.")


if __name__ == "__main__":
    main()
