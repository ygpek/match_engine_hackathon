"""Match Studio AI — Streamlit MVP. Run: streamlit run match_studio/app.py

Reads existing orchestrator JSON and optional Azure narrative JSON; no API calls.
"""

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from historical_visuals import load_history, league_table, interval_history, INTERVALS
from tactical_simulator import load_pre_match_profiles, evaluate_substitution
from player_performance import load_player_history, player_profile, participation, METRICS

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PHASES = ["prematch", "halftime", "live", "postmatch"]
LABELS = {"prematch": "Pre-match", "halftime": "Halftime", "live": "Live", "postmatch": "Post-match"}
HOME_COLOR, AWAY_COLOR = "#16a085", "#7c8aa5"

st.set_page_config(page_title="Match Studio AI", page_icon="⚽", layout="wide")
st.markdown(
    """<style>
.block-container {padding-top:1.8rem; max-width:1350px}
[data-testid="stMetric"] {background:rgba(127,127,127,.07); padding:12px; border-radius:10px}
</style>""",
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def read_json(path: str):
    return json.loads(Path(path).read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def read_players(path: str):
    return pd.read_parquet(path)


def team_name(team_id, lookup):
    return lookup.get(str(team_id), str(team_id))


def num(value, digits=0):
    try:
        return f"{float(value):,.{digits}f}"
    except (ValueError, TypeError):
        return "—"


def compare_bar(label, left, right, left_name, right_name, decimals=0):
    a, b = float(left or 0), float(right or 0)
    denom = max(a + b, 1e-9)
    c1, c2, c3 = st.columns([1, 3, 1])
    c1.markdown(f"**{num(a, decimals)}**")
    c2.markdown(f"<div style='text-align:center;font-size:.83rem'>{label}</div>", unsafe_allow_html=True)
    c2.progress(min(1.0, max(0.0, a / denom)), text=None)
    c3.markdown(f"<div style='text-align:right'><b>{num(b, decimals)}</b></div>", unsafe_allow_html=True)


def extract_options(payload, side):
    side_data = payload.get("tactical_analysis", {}).get(side, {})
    rows = []
    for group in side_data.get("substitution_options_by_position", []):
        for r in group.get("ranked_candidates", []):
            rows.append(
                {
                    "Position": group.get("position"),
                    "Player": r.get("player_name", r.get("player_id")),
                    "Suitability": r.get("suitability_score"),
                    "Attacking": r.get("attacking_score"),
                    "Passing": r.get("passing_score"),
                    "Defending": r.get("defending_score"),
                    "Possession": r.get("possession_score"),
                    "Physical": r.get("physical_score"),
                }
            )
    return pd.DataFrame(rows)


def show_substitution_comparisons(payload, names, home_name, away_name):
    st.caption(
        "Same-position comparisons using pre-match player profiles. "
        "Positive differences indicate higher scenario suitability, not a "
        "predicted performance improvement or win probability."
    )
    for side, team_name in (("home", home_name), ("away", away_name)):
        section = payload.get("tactical_analysis", {}).get(side, {})
        scenario = section.get("scenario", "balanced").replace("_", " ")
        st.markdown(f"#### {team_name} · {scenario.title()}")
        comparisons = section.get("substitution_comparisons", [])
        if not comparisons:
            st.info(
                "No positive, same-position substitution comparison supported "
                "by the available historical player profiles."
            )
            continue
        seen_in, seen_out = set(), set()
        displayed = 0
        for item in comparisons:
            incoming, outgoing = item["in_player_id"], item["out_player_id"]
            if incoming in seen_in or outgoing in seen_out:
                continue
            seen_in.add(incoming)
            seen_out.add(outgoing)
            with st.container(border=True):
                st.markdown(
                    f"**{names.get(outgoing, 'Unknown player')} → "
                    f"{names.get(incoming, 'Unknown player')}** · {item['position']}"
                )
                st.metric("Suitability difference (points)", f"{item['suitability_difference']:+.2f}")
                st.caption(
                    f"Current player: {item['out_suitability']:.1f} · "
                    f"Bench candidate: {item['in_suitability']:.1f} "
                    f"(scenario-relative scores)"
                )
            displayed += 1
            if displayed == 3:
                break


@st.cache_data(show_spinner=False)
def cached_simulator_profiles(match_id, match_date, home_id, away_id):
    return load_pre_match_profiles(
        {
            "match_id": match_id,
            "date": match_date,
            "home_team_id": home_id,
            "away_team_id": away_id,
        }
    )


def show_tactical_simulator(payload, names, positions, home_name, away_name):
    st.markdown("### Interactive tactical simulator")
    st.caption(
        "Choose an on-field player, an eligible bench replacement and a tactical objective. "
        "Scores use only pre-match history; this does not simulate the match result."
    )
    side = st.radio(
        "Team",
        ["home", "away"],
        horizontal=True,
        format_func=lambda s: home_name if s == "home" else away_name,
        key="sim_team",
    )
    section = payload.get("tactical_analysis", {}).get(side, {})
    scenarios = {
        "balanced": "Balanced",
        "protect_lead": "Protect the lead",
        "chase_goal": "Chase a goal",
        "control_midfield": "Control midfield",
        "high_press": "High press",
        "counterattack": "Counterattack",
    }
    default = section.get("scenario", "balanced")
    scenario = st.selectbox(
        "Tactical objective",
        list(scenarios),
        index=list(scenarios).index(default) if default in scenarios else 0,
        format_func=lambda s: scenarios[s],
        key=f"sim_scenario_{side}",
    )
    active = sorted((str(x) for x in section.get("active_player_ids", [])), key=lambda pid: names.get(pid, pid))
    bench = sorted((str(x) for x in section.get("available_bench_player_ids", [])), key=lambda pid: names.get(pid, pid))
    outfield = [pid for pid in active if positions.get(pid) not in (None, "GK")]
    if not outfield:
        st.info("No eligible outfield players in this phase.")
        return
    left, right = st.columns(2)
    with left:
        outgoing = st.selectbox(
            "Player off",
            outfield,
            format_func=lambda pid: f"{names.get(pid, pid)} · {positions.get(pid, '')}",
            key=f"sim_out_{side}",
        )
    compatible = [pid for pid in bench if positions.get(pid) == positions.get(outgoing)]
    with right:
        if not compatible:
            st.info("No unused bench player in this position.")
            return
        incoming = st.selectbox(
            "Player on",
            compatible,
            format_func=lambda pid: f"{names.get(pid, pid)} · {positions.get(pid, '')}",
            key=f"sim_in_{side}_{outgoing}",
        )
    fixture = payload["fixture"]
    try:
        profiles = cached_simulator_profiles(
            str(fixture["match_id"]),
            str(fixture["date"]),
            str(fixture["home_team_id"]),
            str(fixture["away_team_id"]),
        )
        result = evaluate_substitution(profiles, payload, side, outgoing, incoming, scenario)
    except (ValueError, FileNotFoundError, KeyError, IndexError) as exc:
        st.warning(f"Comparison unavailable: {exc}")
        return
    c1, c2, c3 = st.columns(3)
    c1.metric("Current player suitability", f"{result['out_score']:.2f}")
    c2.metric("Replacement suitability", f"{result['in_score']:.2f}")
    c3.metric("Suitability difference", f"{result['difference']:+.2f}")
    st.caption(
        "A positive difference indicates a higher historical position-relative score "
        "for the chosen objective, not an expected increase in goals or win probability."
    )
    with st.expander("Category breakdown"):
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Category": k.title(),
                        names.get(outgoing, "Current"): round(v[0], 2),
                        names.get(incoming, "Replacement"): round(v[1], 2),
                    }
                    for k, v in result["categories"].items()
                ]
            ),
            hide_index=True,
            use_container_width=True,
        )


@st.cache_data(show_spinner=False)
def cached_player_history(match_id, date):
    return load_player_history({"match_id": match_id, "date": date})


@st.cache_data(show_spinner=False)
def cached_featured_lineups():
    return pd.read_parquet(ROOT / "featured_match" / "data" / "featured_match_lineups.parquet")


def show_player_performance(payload, names, positions, home_name, away_name):
    st.markdown("### Individual player performance")
    st.caption(
        "Historical metrics are per 90 minutes and exclude the featured match and all later matches. "
        "Featured-match individual actions cannot be attributed because shot events have no shooter IDs."
    )
    fixture = payload["fixture"]
    try:
        history = cached_player_history(str(fixture["match_id"]), str(fixture["date"]))
        lineups = cached_featured_lineups()
    except (FileNotFoundError, KeyError, ValueError) as exc:
        st.warning(f"Player analysis unavailable: {exc}")
        return
    side = st.radio(
        "Team",
        ["home", "away"],
        horizontal=True,
        format_func=lambda x: home_name if x == "home" else away_name,
        key="performance_team_side_v2",
    )
    # Normalize unexpected values from older Streamlit widget state.
    if side not in ("home", "away"):
        if side == home_name:
            side = "home"
        elif side == away_name:
            side = "away"
        else:
            st.error("Invalid team selection. Refresh the page to reset the selector.")
            return
    team_id = str(fixture[f"{side}_team_id"])
    team_players = lineups.loc[lineups.team_id.astype(str) == team_id, "player_id"].astype(str).tolist()
    if not team_players:
        st.info("No featured-match participants available.")
        return
    player_id = st.selectbox(
        "Player",
        sorted(team_players, key=lambda x: names.get(x, x)),
        format_func=lambda x: f'{names.get(x, x)} · {positions.get(x, "?")}',
        key=f"performance_player_{side}",
    )
    sample = st.select_slider(
        "Historical sample (last appearances)", options=[5, 10, 20, 50], value=10, key="performance_sample"
    )
    profile = player_profile(history, player_id, last_n=sample)
    minute = int(payload.get("state", {}).get("minute", 0) or 0)
    current = participation(lineups, player_id, minute)
    c1, c2, c3 = st.columns(3)
    c1.metric("Historical appearances", profile["matches"])
    c2.metric("Historical minutes", profile["minutes"])
    c3.metric("Observed match minutes", current["observed_minutes"] if current else "—")
    if current:
        if minute == 0:
            st.caption("Pre-match: participation has not started.")
        elif current["currently_on_pitch"]:
            st.caption("Currently on the pitch · " + ("Starter" if current["started"] else "Substitute"))
        elif current["minute_off"] is not None:
            st.caption(f'Substituted off at minute {current["minute_off"]}.')
        else:
            st.caption("Not yet on the pitch at this snapshot.")
    if not profile["metrics"]:
        st.info("No eligible historical player statistics for this sample.")
        return
    st.markdown("#### Historical performance per 90")
    metrics = profile["metrics"]
    st.dataframe(
        pd.DataFrame([{"Metric": key, "Value": val} for key, val in metrics.items()]),
        hide_index=True,
        use_container_width=True,
    )
    st.caption(
        "Pass completion is a percentage, not a per-90 rate. Historical physical metrics "
        "are simulated performance observations, not evidence of current fatigue."
    )
    st.markdown("#### Compare with a teammate")
    others = [pid for pid in team_players if pid != player_id and positions.get(pid) == positions.get(player_id)]
    if others:
        other = st.selectbox(
            "Comparison player (same position)",
            sorted(others, key=lambda x: names.get(x, x)),
            format_func=lambda x: names.get(x, x),
            key=f"performance_compare_{side}_{player_id}",
        )
        second = player_profile(history, other, last_n=sample)
        rows = [
            {
                "Metric": metric,
                names.get(player_id, player_id): value,
                names.get(other, other): second["metrics"].get(metric),
            }
            for metric, value in metrics.items()
        ]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.caption(
            f'Comparison player: {second["matches"]} historical appearances, '
            f'{second["minutes"]} minutes in the selected sample.'
        )
    else:
        st.info("No same-position teammate appeared in the featured match.")


def render_report(report, phase):
    content = report.get("report", report)
    st.subheader(content.get("headline", "AI match analysis"))
    st.write(content.get("overview", ""))
    st.markdown("#### Key insights")
    for item in content.get("key_insights", []):
        st.markdown(f"- {item.get('text', '')}")
    if content.get("tactical_options") and phase in ("halftime", "live"):
        st.markdown("#### Substitution analysis")
        for item in content["tactical_options"]:
            st.markdown(f"- {item.get('text', '')}")


@st.cache_data(show_spinner=False)
def load_roster(path):
    return pd.read_parquet(path)


def roster_maps():
    path = ROOT / "data_generator" / "data" / "players.parquet"
    if not path.exists():
        return {}, {}
    df = load_roster(str(path))
    names = dict(zip(df.player_id.astype(str), df.player_name.astype(str)))
    positions = dict(zip(df.player_id.astype(str), df.primary_position.astype(str)))
    return names, positions


def pitch_figure(player_ids, names, positions, starters):
    """Illustrative positional layout, not measured tracking coordinates."""
    groups = {p: [] for p in ("GK", "CB", "WB", "CM", "AMW", "ST")}
    for pid in player_ids:
        groups.setdefault(positions.get(pid, "CM"), []).append(pid)
    # Broad positional bands. Slots are illustrative; there is no tracking data.
    bands = {"GK": 9, "CB": 26, "WB": 35, "CM": 51, "AMW": 72, "ST": 87}
    fig = go.Figure()
    fig.add_shape(
        type="rect",
        x0=0,
        y0=0,
        x1=100,
        y1=100,
        line=dict(color="white", width=2),
        fillcolor="#176847",
        layer="below",
    )
    fig.add_shape(
        type="rect",
        x0=0,
        y0=0,
        x1=100,
        y1=50,
        line=dict(color="white", width=1),
        fillcolor="#1d754f",
        opacity=0.3,
        layer="below",
    )
    fig.add_shape(
        type="line",
        x0=0,
        x1=100,
        y0=50,
        y1=50,
        line=dict(color="white", width=1),
        layer="below",
    )
    fig.add_shape(
        type="circle",
        x0=37,
        x1=63,
        y0=37,
        y1=63,
        line=dict(color="white", width=1),
        layer="below",
    )
    fig.add_shape(
        type="rect",
        x0=20,
        x1=80,
        y0=0,
        y1=15,
        line=dict(color="white", width=1),
        layer="below",
    )
    fig.add_shape(
        type="rect",
        x0=20,
        x1=80,
        y0=85,
        y1=100,
        line=dict(color="white", width=1),
        layer="below",
    )
    for position, ids in groups.items():
        if not ids:
            continue
        # Players are distributed across their positional band, not actual formations.
        if position == "GK":
            xs = [50]
        elif position == "CB":
            xs = {
                1: [50],
                2: [40, 60],
                3: [30, 50, 70],
                4: [25, 42, 58, 75],
            }.get(len(ids), [30 + 40 * i / (len(ids) - 1) for i in range(len(ids))])
        elif position == "WB":
            xs = (
                [12, 88]
                if len(ids) == 2
                else ([50] if len(ids) == 1 else [12 + 76 * i / (len(ids) - 1) for i in range(len(ids))])
            )
        else:
            xs = [50] if len(ids) == 1 else [18 + 64 * i / (len(ids) - 1) for i in range(len(ids))]
        ys = [bands.get(position, 51)] * len(ids)
        labels = [names.get(pid, "Unknown player") for pid in ids]
        statuses = ["Starter" if pid in starters else "Substitute on" for pid in ids]
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="markers+text",
                name=position,
                marker=dict(
                    size=23,
                    color=["#f4f7fb" if pid in starters else "#ffd166" for pid in ids],
                    line=dict(color="#163f35", width=2),
                ),
                text=[n.split()[-1][:15] for n in labels],
                textposition="top center",
                textfont=dict(color="white", size=11),
                customdata=[[n, position, status] for n, status in zip(labels, statuses)],
                hovertemplate="%{customdata[0]}<br>%{customdata[1]} · %{customdata[2]}<extra></extra>",
            )
        )
    fig.update_layout(
        height=540,
        margin=dict(l=5, r=5, t=5, b=5),
        showlegend=False,
        paper_bgcolor="#176847",
        plot_bgcolor="#176847",
        xaxis=dict(range=[-4, 104], visible=False, fixedrange=True),
        yaxis=dict(range=[-4, 104], visible=False, fixedrange=True, scaleanchor="x"),
    )
    return fig


def show_lineups(payload, home_name, away_name, home_id, away_id, names, positions):
    st.caption(
        "Illustrative pitch placement by roster position — not tracked player coordinates or an inferred formation. Yellow markers indicate substitutes who have entered."
    )
    active = payload.get("state", {}).get("active_player_ids", {})
    lineups_path = ROOT / "featured_match" / "data" / "featured_match_lineups.parquet"
    starters_by_team = {}
    if lineups_path.exists():
        lineup_df = load_roster(str(lineups_path))
        for tid in (home_id, away_id):
            starters_by_team[tid] = set(
                lineup_df.loc[
                    (lineup_df.team_id.astype(str) == tid) & lineup_df.started.astype(bool), "player_id"
                ].astype(str)
            )
    c1, c2 = st.columns(2)
    for col, tid, team in [(c1, home_id, home_name), (c2, away_id, away_name)]:
        with col:
            st.markdown(f"#### {team}")
            ids = [str(x) for x in active.get(tid, [])]
            if not ids and payload.get("phase") == "prematch":
                ids = sorted(starters_by_team.get(tid, set()))
            if not ids:
                st.info("No lineup available for this phase.")
                continue
            starters = starters_by_team.get(tid, set(ids))
            st.plotly_chart(
                pitch_figure(ids, names, positions, starters),
                use_container_width=True,
                key=f"pitch_{tid}_{payload.get('phase')}",
            )
            options = sorted(ids, key=lambda pid: names.get(pid, pid))
            chosen = st.selectbox(
                "Player profile",
                options,
                format_func=lambda pid: names.get(pid, "Unknown player"),
                key=f"profile_{tid}_{payload.get('phase')}",
            )
            if chosen:
                st.markdown(f"**{names.get(chosen, 'Unknown player')}** · {positions.get(chosen, 'Unknown position')}")
                st.caption("Starting XI" if chosen in starters else "Entered as substitute")
                # Display available, grounded scenario-relative comparison data.
                section = payload.get("tactical_analysis", {}).get("home" if tid == home_id else "away", {})
                related = [
                    x
                    for x in section.get("substitution_comparisons", [])
                    if x.get("out_player_id") == chosen or x.get("in_player_id") == chosen
                ]
                if related:
                    top = related[0]
                    st.caption(
                        f"Scenario: {top.get('scenario', 'balanced').replace('_', ' ')} · "
                        f"Substitution suitability difference: {top['suitability_difference']:+.2f}"
                    )
            removed = sorted(starters - set(ids))
            if removed:
                st.caption("Substituted off: " + ", ".join(names.get(pid, "Unknown player") for pid in removed))


def show_form(label, record):
    if not isinstance(record, dict) or not record.get("sample_size"):
        return
    st.markdown(f"#### {label}")
    cols = st.columns(4)
    for col, (key, title) in zip(
        cols, [("wins", "Wins"), ("draws", "Draws"), ("losses", "Losses"), ("points", "Points")]
    ):
        col.metric(title, record.get(key, "—"))
    form = record.get("record_oldest_to_newest", "")
    if form:
        st.caption(f"Last {record['sample_size']} matches (oldest → newest): " + " · ".join(form))
    metrics = {
        "Goals for": record.get("goals_for"),
        "Goals against": record.get("goals_against"),
        "xG / match": record.get("xg_for_per_match"),
        "Shots / match": record.get("shots_per_match"),
        "Possession %": record.get("possession_pct_mean"),
    }
    st.dataframe(
        pd.DataFrame([{"Metric": k, "Value": v} for k, v in metrics.items() if v is not None]),
        hide_index=True,
        use_container_width=True,
    )


def show_history(context, home_name, away_name):
    st.caption("Only matches before the featured fixture are used for historical comparisons.")
    a, b = st.columns(2)
    with a:
        show_form(f"{home_name} · Recent form", context.get("recent_form"))
        show_form("Home matches", context.get("home_form"))
    with b:
        show_form(f"{away_name} · Recent form", context.get("opponent_recent_form"))
    patterns = context.get("halftime_patterns", {})
    rows = []
    for state, info in patterns.items():
        if isinstance(info, dict) and info.get("sample_size"):
            rows.append(
                {
                    "Halftime state": state.title(),
                    "Matches": info["sample_size"],
                    "W": info.get("full_time_wins", 0),
                    "D": info.get("full_time_draws", 0),
                    "L": info.get("full_time_losses", 0),
                    "Win rate %": info.get("full_time_win_rate_pct"),
                }
            )
    if rows:
        st.markdown("#### Historical halftime outcomes · Home team")
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    h2h = context.get("head_to_head", {})
    if h2h and h2h.get("sample_size"):
        st.markdown("#### Head-to-head · Home team perspective")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Matches": h2h["sample_size"],
                        "Wins": h2h.get("wins", 0),
                        "Draws": h2h.get("draws", 0),
                        "Losses": h2h.get("losses", 0),
                    }
                ]
            ),
            hide_index=True,
            use_container_width=True,
        )


@st.cache_data(show_spinner=False)
def historical_files(data_dir):
    return load_history(Path(data_dir))


def show_league(payload, phase, home_id, away_id):
    data_dir = ROOT / "data_generator" / "data"
    try:
        matches, shots, teams = historical_files(str(data_dir))
        table = league_table(matches, teams, payload["fixture"], phase)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        st.info(f"League table unavailable: {exc}")
        return
    st.markdown("#### League standings" + (" · after featured match" if phase == "postmatch" else " · before kickoff"))
    st.caption(
        "Current season only. Rankings use points, goal difference, then goals scored; teams tied on all three are alphabetically ordered."
    )
    display = table.drop(columns="_team_id")
    featured_names = set(table.loc[table._team_id.astype(str).isin([str(home_id), str(away_id)]), "Team"])

    def highlight_featured(row):
        if row["Team"] in featured_names:
            return ["background-color: rgba(22, 160, 133, 0.20); font-weight: bold"] * len(row)
        return [""] * len(row)

    st.dataframe(
        display.style.apply(highlight_featured, axis=1),
        hide_index=True,
        use_container_width=True,
        column_config={"Pos": st.column_config.NumberColumn(width="small")},
    )
    selected = table[table._team_id.isin([home_id, away_id])]
    if len(selected) == 2:
        st.caption("Featured teams: " + " · ".join(f"{r.Team}: {r.Pos} ({r.Pts} pts)" for r in selected.itertuples()))


def show_interval_comparison(payload, home_id, away_id, home_name, away_name):
    data_dir = ROOT / "data_generator" / "data"
    try:
        matches, shots, teams = historical_files(str(data_dir))
    except (FileNotFoundError, KeyError, ValueError) as exc:
        st.info(f"Interval analysis unavailable: {exc}")
        return

    c1, c2 = st.columns(2)
    with c1:
        last_n = st.selectbox("Previous league matches", [5, 10, 15, 20], index=1, key="interval_last_n")
    with c2:
        measure = st.selectbox(
            "Compare across intervals",
            [
                "Goal difference",
                "xG difference",
                "Goals for",
                "Goals against",
                "xG",
                "xGA",
                "Shots for",
                "Shots against",
                "Goals - xG",
            ],
            index=1,
            key="interval_measure",
        )
    try:
        history = interval_history(matches, shots, payload["fixture"], [home_id, away_id], last_n=last_n)
    except (KeyError, ValueError) as exc:
        st.info(f"Interval analysis unavailable: {exc}")
        return
    if history.empty or history.Matches.max() == 0:
        st.info("No historical matches available before this fixture.")
        return

    minute = int(payload.get("state", {}).get("minute", 67) or 67)
    index = next((i for i, (a, b) in enumerate(INTERVALS) if a < minute <= b), len(INTERVALS) - 1)
    labels = [f"{a}–{b}" for a, b in INTERVALS]
    selected = st.selectbox("Inspect 15-minute interval", labels, index=index, key="interval_selected")
    sub = history.loc[history.Interval == selected]
    if len(sub) < 2:
        st.info("Insufficient interval data for both teams.")
        return
    a = sub.loc[sub.team_id.astype(str) == str(home_id)].iloc[0]
    b = sub.loc[sub.team_id.astype(str) == str(away_id)].iloc[0]
    metrics = [
        "Matches",
        "Shots for",
        "Shots against",
        "Goals for",
        "Goals against",
        "Goal difference",
        "xG",
        "xGA",
        "xG difference",
        "Goals - xG",
        "Goals against - xGA",
        "xG per match",
        "xGA per match",
    ]
    st.dataframe(
        pd.DataFrame([{"Metric": m, home_name: a[m], away_name: b[m]} for m in metrics]),
        hide_index=True,
        use_container_width=True,
    )
    st.caption(
        "Historical totals are calculated separately from each team's last N "
        "league matches in the featured season, strictly before kickoff. "
        "xGA is opponents' shot xG. Goals − xG is a difference, not a ratio. "
        "Per-match metrics divide the interval total by available matches."
    )

    plot = history.pivot(index="Interval", columns="team_id", values=measure).reindex(labels)
    fig = go.Figure()
    for tid, name, color in [(home_id, home_name, HOME_COLOR), (away_id, away_name, AWAY_COLOR)]:
        if str(tid) in plot.columns:
            fig.add_trace(go.Bar(name=name, x=labels, y=plot[str(tid)].tolist(), marker_color=color))
    fig.update_layout(
        barmode="group",
        height=330,
        margin=dict(l=10, r=10, t=20, b=10),
        xaxis_title="Match interval (minutes)",
        yaxis_title=f"{measure} · total across previous matches",
    )
    st.plotly_chart(fig, use_container_width=True, key="interval_comparison_chart")
    st.caption(
        "This is descriptive historical evidence, not a prediction for "
        "the current match or a measure of individual player fatigue."
    )


st.title("⚽ Match Studio AI")
st.caption("Football match intelligence · Synthetic demonstration")

available = {p: HERE / "output" / f"match_studio_{p}.json" for p in PHASES}
missing = [p for p, path in available.items() if not path.exists()]
if len(missing) == len(PHASES):
    st.error("No orchestration outputs found. Run `python match_studio/match_studio_orchestrator.py` first.")
    st.stop()

with st.sidebar:
    st.header("Match controls")
    valid = [p for p in PHASES if available[p].exists()]
    phase = st.radio("Match phase", valid, format_func=lambda p: LABELS[p], horizontal=False)
    if st.button("Refresh files"):
        st.cache_data.clear()
        st.rerun()
    st.caption("This dashboard reads local files only. It does not call Azure or regenerate data.")

payload = read_json(str(available[phase]))
fixture = payload.get("fixture", {})
state = payload.get("state", {})
home_id = str(fixture.get("home_team_id", "home"))
away_id = str(fixture.get("away_team_id", "away"))
team_lookup = {}
team_path = ROOT / "data_generator" / "data" / "teams.parquet"
if team_path.exists():
    teams = read_players(str(team_path))
    name_col = next((c for c in ("team_name", "name") if c in teams.columns), None)
    if name_col and "team_id" in teams.columns:
        team_lookup = dict(zip(teams.team_id.astype(str), teams[name_col].astype(str)))
home_name, away_name = team_name(home_id, team_lookup), team_name(away_id, team_lookup)

player_names, player_positions = roster_maps()
st.caption(f"{fixture.get('date', 'Featured match')} · {LABELS[phase]} · {state.get('minute', 0)}′")
left, center, right = st.columns([3, 2, 3], vertical_alignment="center")
left.markdown(f"### {home_name}")
if phase == "prematch":
    center.markdown("<h1 style='text-align:center'>vs</h1>", unsafe_allow_html=True)
else:
    center.markdown(
        f"<h1 style='text-align:center'>{state.get('home_goals', 0)} – {state.get('away_goals', 0)}</h1>",
        unsafe_allow_html=True,
    )
right.markdown(f"<h3 style='text-align:right'>{away_name}</h3>", unsafe_allow_html=True)

tab_names = ["📊 Match statistics", "👥 Lineups", "👤 Player performance", "✨ AI analysis", "📚 Historical context"]
if phase in ("halftime", "live"):
    tab_names.insert(4, "🔁 Substitution candidates")
tabs = st.tabs(tab_names)
stats_tab, lineup_tab, player_tab, analysis_tab = tabs[:4]
if phase in ("halftime", "live"):
    tactics_tab, history_tab = tabs[4:]
else:
    tactics_tab, history_tab = None, tabs[4]
with lineup_tab:
    show_lineups(payload, home_name, away_name, home_id, away_id, player_names, player_positions)
with player_tab:
    show_player_performance(payload, player_names, player_positions, home_name, away_name)
with stats_tab:
    if phase == "prematch":
        st.info(
            "Match statistics become available after kickoff. Historical information is in the Historical context tab."
        )
    else:
        home_shots = state.get("shots", {}).get(home_id, {})
        away_shots = state.get("shots", {}).get(away_id, {})
        st.markdown("#### Attacking")
        for label, key, decimals in [
            ("Shots", "shots", 0),
            ("Shots on target", "shots_on_target", 0),
            ("Expected goals (xG)", "xg", 2),
        ]:
            compare_bar(label, home_shots.get(key, 0), away_shots.get(key, 0), home_name, away_name, decimals)
        home_half = state.get("completed_half_stats", {}).get(home_id, {})
        away_half = state.get("completed_half_stats", {}).get(away_id, {})
        if int(home_half.get("complete_through_minute", 0)) > 0:
            st.markdown("#### Completed-half statistics")
            st.caption(f"Only completed halves through minute {home_half['complete_through_minute']} are included.")
            for label, key in [
                ("Passes attempted", "passes_attempted"),
                ("Passes completed", "passes_completed"),
                ("Tackles", "tackles"),
                ("Interceptions", "interceptions"),
                ("Recoveries", "recoveries"),
                ("Corners", "corners"),
            ]:
                compare_bar(label, home_half.get(key, 0), away_half.get(key, 0), home_name, away_name)
        events = pd.DataFrame(payload.get("observed_events", []))
        if not events.empty and {"minute", "event_type", "team_id"}.issubset(events.columns):
            st.markdown("#### Shot timeline")
            shots = events[events.event_type.isin(["shot", "goal"])].copy()
            if not shots.empty:
                shots["team"] = shots.team_id.astype(str).map({home_id: home_name, away_id: away_name})
                shots["xg"] = pd.to_numeric(shots.get("xg", 0), errors="coerce").fillna(0)
                fig = go.Figure()
                for team, color in [(home_name, HOME_COLOR), (away_name, AWAY_COLOR)]:
                    subset = shots[shots.team == team].sort_values("minute")
                    if not subset.empty:
                        fig.add_trace(
                            go.Scatter(
                                x=subset.minute,
                                y=subset.xg.cumsum(),
                                mode="lines+markers",
                                name=team,
                                line=dict(color=color),
                                customdata=subset.event_type,
                                hovertemplate="%{x}′ · Cumulative xG %{y:.2f}<br>%{customdata}<extra></extra>",
                            )
                        )
                fig.update_layout(
                    height=340, margin=dict(l=20, r=20, t=20, b=20), xaxis_title="Minute", yaxis_title="Cumulative xG"
                )
                st.plotly_chart(fig, use_container_width=True)
            st.markdown("#### Match events")
            display = [c for c in ("minute", "team_id", "event_type", "description", "xg") if c in events.columns]
            event_table = events[display].copy()
            if "team_id" in event_table:
                event_table["team_id"] = (
                    event_table["team_id"].astype(str).map({home_id: home_name, away_id: away_name}).fillna("Unknown")
                )
                event_table = event_table.rename(columns={"team_id": "Team"})
            st.dataframe(event_table.sort_values("minute", ascending=False), hide_index=True, use_container_width=True)

with analysis_tab:
    report_path = HERE / "narratives" / f"match_studio_{phase}_report.json"
    if report_path.exists():
        render_report(read_json(str(report_path)), phase)
    else:
        st.info(
            f"No Azure narrative for {LABELS[phase]} yet. Generate it with `python match_studio/azure_narrative_engine.py --phase {phase}`."
        )
        md_path = HERE / "narratives" / f"match_studio_{phase}_report.md"
        if md_path.exists():
            st.markdown(md_path.read_text(encoding="utf-8"))

if tactics_tab is not None:
    with tactics_tab:
        show_tactical_simulator(payload, player_names, player_positions, home_name, away_name)
        st.divider()
        show_substitution_comparisons(payload, player_names, home_name, away_name)


with history_tab:
    if phase in ("prematch", "postmatch"):
        show_league(payload, phase, home_id, away_id)
    if phase == "live":
        st.markdown("#### Historical performance by match interval")
        show_interval_comparison(payload, home_id, away_id, home_name, away_name)
    show_history(payload.get("historical_context", {}), home_name, away_name)

st.divider()
st.caption(
    "All clubs, players and matches are fictional. Data is generated locally; narratives are created separately via Azure AI."
)
