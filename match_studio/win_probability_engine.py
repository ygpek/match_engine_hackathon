"""Leakage-safe, Poisson-based in-play match outcome forecasts."""
from math import exp, factorial
import pandas as pd

def _pmf(rate):
    return [exp(-rate)*rate**i/factorial(i) for i in range(21)]

def _rates(matches, fixture):
    past = matches.loc[(pd.to_datetime(matches.date) < pd.to_datetime(fixture["date"])) &
                       (matches.match_id.astype(str) != str(fixture["match_id"]))].copy()
    if "season" in past and "season" in fixture:
        past = past.loc[past.season.astype(str) == str(fixture["season"])]
    if past.empty:
        return 1.45, 1.15, 0
    hg, ag = max(.2, past.ft_home_goals.mean()), max(.2, past.ft_away_goals.mean())
    home, away = str(fixture["home_team_id"]), str(fixture["away_team_id"])
    def smooth(team, side, col, baseline):
        rows = past.loc[past[side].astype(str) == team]
        return (rows[col].sum() + 8*baseline)/(len(rows)+8)
    home_rate = smooth(home,"home_team_id","ft_home_goals",hg) * smooth(away,"away_team_id","ft_home_goals",hg)/hg
    away_rate = smooth(away,"away_team_id","ft_away_goals",ag) * smooth(home,"home_team_id","ft_away_goals",ag)/ag
    return max(.1,min(5,home_rate)), max(.1,min(5,away_rate)),len(past)

def predict(matches, fixture, minute, home_goals, away_goals):
    minute = max(0,min(90,int(minute)))
    hrate, arate, n = _rates(matches,fixture)
    remaining = (90-minute)/90
    hleft, aleft = hrate*remaining, arate*remaining
    hp,ap = _pmf(hleft),_pmf(aleft)
    hw=dr=aw=0.
    for i,p in enumerate(hp):
        for j,q in enumerate(ap):
            if home_goals+i>away_goals+j: hw+=p*q
            elif home_goals+i==away_goals+j: dr+=p*q
            else: aw+=p*q
    total=hw+dr+aw
    return {"home_win":hw/total,"draw":dr/total,"away_win":aw/total,
            "remaining_home_xg":hleft,"remaining_away_xg":aleft,"historical_sample":n}

def score_at(events, minute, home_id, away_id):
    h=a=0
    for e in events:
        if e.get("event_type")!="goal" or int(e.get("minute") or 0)>minute: continue
        if str(e.get("team_id"))==str(home_id): h+=1
        elif str(e.get("team_id"))==str(away_id): a+=1
    return h,a
