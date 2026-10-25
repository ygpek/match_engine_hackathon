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
from league_player_leaders import league_player_leaders
from player_similarity_ui import similar_players, radar_profiles
from win_probability_engine import predict, score_at

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PHASES = ["prematch", "halftime", "live", "postmatch"]
LABELS = {"prematch": "Pre-match", "halftime": "Halftime", "live": "Live", "postmatch": "Post-match"}
HOME_COLOR, AWAY_COLOR = "#16a085", "#7c8aa5"
POSITION_ORDER = {"GK": 0, "CB": 1, "WB": 2, "CM": 3, "AMW": 4, "CF": 5, "ST": 5}


def player_sort_key(pid, names, positions):
    return (POSITION_ORDER.get(positions.get(str(pid), ""), 99), names.get(str(pid), str(pid)).casefold())


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
    active = sorted(
        (str(x) for x in section.get("active_player_ids", [])), key=lambda pid: player_sort_key(pid, names, positions)
    )
    bench = sorted(
        (str(x) for x in section.get("available_bench_player_ids", [])),
        key=lambda pid: player_sort_key(pid, names, positions),
    )
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
    # GK similarity is not supported; keep goalkeepers available for performance
    # analysis, but select an outfield player (preferably CB) by default.
    sorted_players = sorted(team_players, key=lambda x: player_sort_key(x, names, positions))
    default_player = next((p for p in sorted_players if positions.get(p) == "CB"), None)
    if default_player is None:
        default_player = next((p for p in sorted_players if positions.get(p) != "GK"), sorted_players[0])
    player_id = st.selectbox(
        "Player",
        sorted_players,
        index=sorted_players.index(default_player),
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
    st.markdown("#### Similar players from other clubs")
    top_n = st.select_slider("Number of similar players", options=[3, 5, 10, 15, 20], value=5, key="similar_top_n")
    if positions.get(player_id) == "GK":
        st.info(
            "Goalkeeper similarity is unavailable: the historical data does not include the save and shot-stopping metrics needed for a meaningful comparison."
        )
    else:
        try:
            similar = similar_players(history, player_id, team_id, positions, names, top_n=top_n, last_n=sample)
        except (ValueError, KeyError) as exc:
            st.info(f"Similar-player comparison unavailable: {exc}")
        else:
            if similar.empty:
                st.info("No eligible players from other teams with sufficient historical playing time.")
            else:
                team_path = ROOT / "data_generator" / "data" / "teams.parquet"
                if team_path.exists():
                    teams = read_players(str(team_path))
                    club_names = dict(zip(teams.team_id.astype(str), teams.team_name.astype(str)))
                else:
                    club_names = {}
                similar["Club"] = similar["team_id"].astype(str).map(lambda tid: club_names.get(tid, tid))
                columns = ["Player", "Club", "Position", "Similarity (%)", "Matches", "Minutes"]
                metric_columns = [m for m in metrics if m in similar.columns]
                st.dataframe(similar[columns + metric_columns], hide_index=True, use_container_width=True)
                st.caption(
                    "Similarity compares position-specific per-90 historical statistics over the selected recent appearances, using only matches before kickoff. Scores are relative statistical similarity, not performance ratings or predictions."
                )

                st.markdown("#### Player comparison radar")
                choices = similar.reset_index(drop=True)
                chosen_idx = st.selectbox(
                    "Compare with",
                    range(len(choices)),
                    format_func=lambda i: f"{choices.iloc[i]['Player']} ({choices.iloc[i]['Club']})",
                    key="similar_radar_opponent",
                )
                chosen = choices.iloc[int(chosen_idx)]
                try:
                    radar = radar_profiles(history, player_id, chosen["player_id"], positions, last_n=sample)
                except (ValueError, KeyError) as exc:
                    st.info(f"Radar unavailable: {exc}")
                else:
                    labels = radar["Metric"].tolist()
                    fig = go.Figure()
                    for label, field, color in (
                        (names.get(player_id, player_id), "Reference percentile", "#16a085"),
                        (str(chosen["Player"]), "Comparison percentile", "#e67e22"),
                    ):
                        fig.add_trace(
                            go.Scatterpolar(
                                r=radar[field].tolist() + [float(radar[field].iloc[0])],
                                theta=labels + [labels[0]],
                                fill="toself",
                                name=label,
                                line=dict(color=color),
                                opacity=0.8,
                            )
                        )
                    fig.update_layout(
                        polar=dict(radialaxis=dict(visible=True, range=[0, 100], ticksuffix="th")),
                        showlegend=True,
                        margin=dict(l=55, r=55, t=30, b=30),
                        height=420,
                    )
                    st.plotly_chart(fig, use_container_width=True, key="player_similarity_radar")
                    st.caption(
                        "Axes show percentile ranks (0–100) among same-position players with at least 180 historical minutes, using only pre-match data. Higher means more of that statistic, not necessarily better performance. These are not player ability ratings."
                    )
                    with st.expander("Compare original per-90 values"):
                        st.dataframe(
                            radar[["Metric", "Reference /90", "Comparison /90"]],
                            hide_index=True,
                            use_container_width=True,
                        )


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
                ids = sorted(starters_by_team.get(tid, set()), key=lambda pid: player_sort_key(pid, names, positions))
            if not ids:
                st.info("No lineup available for this phase.")
                continue
            starters = starters_by_team.get(tid, set(ids))
            st.plotly_chart(
                pitch_figure(ids, names, positions, starters),
                use_container_width=True,
                key=f"pitch_{tid}_{payload.get('phase')}",
            )
            options = sorted(ids, key=lambda pid: player_sort_key(pid, names, positions))
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
            dismissed_ids = {
                str(event.get("player_id"))
                for event in payload.get("observed_events", [])
                if event.get("event_type") in ("red_card", "second_yellow_red")
                and str(event.get("team_id")) == str(tid)
            }
            if dismissed_ids:
                st.error("🟥 Sent off: " + ", ".join(names.get(pid, "Unknown player") for pid in sorted(dismissed_ids)))
            removed = sorted(starters - set(ids) - dismissed_ids)
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


def replay_view(payload, minute, home_id, away_id):
    """Derive UI-only state from phase-visible events, never from future match data."""
    import copy

    view = copy.deepcopy(payload)
    events = [
        e for e in payload.get("observed_events", []) if isinstance(e, dict) and int(e.get("minute") or 0) <= minute
    ]
    view["observed_events"] = events
    original = payload.get("state", {})
    state = copy.deepcopy(original)
    state["minute"] = minute
    shot_events = [e for e in events if e.get("event_type") in ("shot", "goal")]
    for side, tid in (("home", home_id), ("away", away_id)):
        subset = [e for e in shot_events if str(e.get("team_id")) == tid]
        state[f"{side}_goals"] = sum(e.get("event_type") == "goal" for e in subset)
        state.setdefault("shots", {})[tid] = {
            "shots": len(subset),
            "shots_on_target": sum(e.get("event_type") == "goal" or e.get("is_on_target") is True for e in subset),
            "xg": round(sum(float(e.get("xg") or 0) for e in subset), 3),
            "goals": state[f"{side}_goals"],
        }
    # Half aggregates are only meaningful after a half has completed. The
    # original phase snapshot contains only aggregates already observed.
    if minute < 45:
        state["completed_half_stats"] = {}
    elif minute < 90 and int(original.get("minute", 0)) == 90:
        # Postmatch snapshot has full-time aggregates. Reconstruct first-half
        # totals from the authoritative generated period data, if available.
        path = ROOT / "featured_match" / "data" / "featured_match_team_half_stats.parquet"
        if path.exists():
            halves = read_players(str(path))
            first = halves[pd.to_numeric(halves.period, errors="coerce") == 1]
            aggregates = {}
            for tid in (home_id, away_id):
                sub = first[first.team_id.astype(str) == tid]
                aggregates[tid] = {
                    k: int(pd.to_numeric(sub[k], errors="coerce").fillna(0).sum())
                    for k in (
                        "passes_attempted",
                        "passes_completed",
                        "tackles",
                        "interceptions",
                        "recoveries",
                        "corners",
                    )
                    if k in sub
                }
                aggregates[tid]["complete_through_minute"] = 45
            state["completed_half_stats"] = aggregates
        else:
            state["completed_half_stats"] = {}
    # Lineups reflect substitutions and dismissals through the selected minute.
    path = ROOT / "featured_match" / "data" / "featured_match_lineups.parquet"
    if path.exists():
        lineups = read_players(str(path))
        if "match_id" in lineups.columns:
            lineups = lineups[lineups.match_id.astype(str) == str(payload.get("fixture", {}).get("match_id"))]
        if minute == 0:
            active = lineups[lineups.started.astype(bool)]
        else:
            active = lineups[
                (lineups.minute_on <= minute)
                & ((lineups.minute_off > minute) | ((minute == 90) & (lineups.minute_off == 90)))
            ]
        state["active_player_ids"] = {
            tid: active.loc[active.team_id.astype(str) == tid, "player_id"].astype(str).tolist()
            for tid in (home_id, away_id)
        }
    view["state"] = state
    return view


def show_league_position_impact(payload):
    """Postmatch movement from the phase-safe league expectations payload."""
    if payload.get("phase") != "postmatch":
        return
    league = payload.get("league_expectations") or {}
    sides = league.get("teams") or {}
    if not all(
        isinstance(sides.get(side), dict)
        and isinstance(sides[side].get("before_match"), dict)
        and isinstance(sides[side].get("after_match"), dict)
        for side in ("home", "away")
    ):
        st.info("League position impact is unavailable for this fixture.")
        return

    st.markdown("#### League position impact")
    st.caption(
        "Positions and points immediately before and after the featured match. "
        "Only this match is added; other fixtures played on the same date are excluded."
    )
    cols = st.columns(2)
    for side, col in zip(("home", "away"), cols):
        item = sides[side]
        before, after = item["before_match"], item["after_match"]
        old_pos, new_pos = int(before["position"]), int(after["position"])
        old_pts, new_pts = int(before["points"]), int(after["points"])
        delta = old_pos - new_pos
        if delta > 0:
            movement = f"↑ Up {delta} position{'s' if delta != 1 else ''}"
        elif delta < 0:
            movement = f"↓ Down {abs(delta)} position{'s' if delta != -1 else ''}"
        else:
            movement = "→ Position unchanged"
        with col:
            st.markdown(f"**{item.get('team_name', side.title())}**")
            st.metric("League position", f"{new_pos}", delta=movement, delta_color="off")
            st.caption(
                f"Position: {old_pos} → {new_pos}  ·  Points: {old_pts} → {new_pts} "
                f"({new_pts-old_pts:+d})  ·  Goal difference: "
                f"{int(before['goal_difference']):+d} → {int(after['goal_difference']):+d}"
            )
    st.divider()


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


def show_league_player_leaders(payload, phase):
    st.markdown("### Individual league leaders")
    st.caption(
        "Current season through the featured fixture: pre-match totals for pre-match, "
        "halftime and live; full-match totals for post-match."
    )
    try:
        leaders = league_player_leaders(ROOT / "data_generator" / "data", payload["fixture"], phase)
    except (FileNotFoundError, KeyError, ValueError, TypeError) as exc:
        st.info(f"Player leaderboards unavailable: {exc}")
        return
    sections = [
        ("Top 5 goalscorers", "goals"),
        ("Top 5 assist providers", "assists"),
        ("Top 5 defenders", "defenders"),
        ("Top 5 goalkeepers", "keepers"),
    ]
    fixture = payload["fixture"]
    featured_ids = {str(fixture["home_team_id"]), str(fixture["away_team_id"])}
    teams = pd.read_parquet(ROOT / "data_generator" / "data" / "teams.parquet")
    featured_clubs = set(teams.loc[teams.team_id.astype(str).isin(featured_ids), "team_name"].astype(str))

    def highlight_featured_player(row):
        if str(row["Club"]) in featured_clubs:
            return ["background-color: rgba(22, 160, 133, 0.20); font-weight: bold"] * len(row)
        return [""] * len(row)

    for title, key in sections:
        st.markdown(f"#### {title}")
        table = leaders[key]
        if table.empty:
            st.info("No qualifying players yet.")
        else:
            st.dataframe(
                table.style.apply(highlight_featured_player, axis=1),
                hide_index=True,
                use_container_width=True,
            )
    st.caption(
        "Defensive actions per game = (tackles + interceptions + recoveries + clearances) "
        "/ appearances; defenders are CB/WB with at least three appearances. "
        "Goalkeeper clean sheets require a full 90 minutes and zero goals conceded."
    )


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
# Replay is a display-only view. Tactical recommendations, player history,
# league standings and AI reports remain tied to the selected official phase.
if phase != "prematch":
    phase_limit = min(90, max(0, int(state.get("minute", 0) or 0)))
    # Streamlit widget state is updated before the script reruns, so the
    # scoreboard can reflect a slider rendered beneath it.
    replay_minute = min(phase_limit, max(0, int(st.session_state.get(f"replay_{phase}", phase_limit))))
    replay_payload = replay_view(payload, replay_minute, home_id, away_id)
    replay_state = replay_payload["state"]
else:
    replay_payload = payload
    replay_state = state
st.caption(f"{fixture.get('date', 'Featured match')} · {LABELS[phase]} · {replay_state.get('minute', 0)}′")
left, center, right = st.columns([3, 2, 3], vertical_alignment="center")
left.markdown(f"### {home_name}")
if phase == "prematch":
    center.markdown("<h1 style='text-align:center'>vs</h1>", unsafe_allow_html=True)
else:
    center.markdown(
        f"<h1 style='text-align:center'>{replay_state.get('home_goals', 0)} – {replay_state.get('away_goals', 0)}</h1>",
        unsafe_allow_html=True,
    )
right.markdown(f"<h3 style='text-align:right'>{away_name}</h3>", unsafe_allow_html=True)

# Goal scorers beneath the scoreboard, using only events observed in this phase.
# Some historical replays do not attribute shots to players; do not invent names.
if phase != "prematch":
    goal_events = [
        event
        for event in replay_payload.get("observed_events", [])
        if isinstance(event, dict) and event.get("event_type") == "goal"
    ]
    goal_events.sort(key=lambda event: (int(event.get("minute") or 0), int(event.get("second") or 0)))

    def scorer_lines(team_id):
        lines = []
        for event in goal_events:
            if str(event.get("team_id")) != str(team_id):
                continue
            player_id = event.get("player_id")
            player_name = player_names.get(str(player_id)) if player_id is not None else None
            scorer = player_name or "Scorer unavailable"
            minute = event.get("minute")
            minute_text = f"{int(minute)}′" if minute is not None else ""
            lines.append(f"{scorer} {minute_text}".strip())
        return lines

    home_scorers, away_scorers = scorer_lines(home_id), scorer_lines(away_id)
    if home_scorers or away_scorers:
        with center:
            for item in home_scorers:
                st.markdown(
                    f"<div style='text-align:center;font-size:.83rem'>⚽ {item} ({home_name})</div>",
                    unsafe_allow_html=True,
                )
            for item in away_scorers:
                st.markdown(
                    f"<div style='text-align:center;font-size:.83rem'>⚽ {item} ({away_name})</div>",
                    unsafe_allow_html=True,
                )


# Replay controls sit beneath the score and goalscorers, above the tabs.
if phase != "prematch":
    st.markdown("#### Interactive match replay")
    if phase_limit:
        st.slider("Replay minute", min_value=0, max_value=phase_limit, value=phase_limit, step=1, key=f"replay_{phase}")
    st.caption(
        f"Explore observed events through {replay_minute}′. "
        f"This {LABELS[phase].lower()} report contains no events after {phase_limit}′. "
        "AI analysis and tactical recommendations remain at the official phase snapshot."
    )


# Historical-only predictive analytics, synchronized with replay minute.
historical_path = ROOT / "data_generator" / "data" / "historical_matches.parquet"
if historical_path.exists():
    try:
        history_matches = read_players(str(historical_path))
        observed = payload.get("observed_events", [])
        selected_minute = replay_minute if phase != "prematch" else 0
        goals = score_at(observed, selected_minute, home_id, away_id)
        prediction = predict(history_matches, fixture, selected_minute, *goals)
        with st.container(border=True):
            st.markdown("#### 📈 Match outcome probability")
            cols = st.columns(3)
            for col, title, key in zip(
                cols, [f"{home_name} win", "Draw", f"{away_name} win"], ["home_win", "draw", "away_win"]
            ):
                col.metric(title, f"{prediction[key]:.1%}")
            st.caption(
                f"Poisson forecast at {selected_minute}′ · Expected remaining goals: "
                f"{home_name} {prediction['remaining_home_xg']:.2f}, "
                f"{away_name} {prediction['remaining_away_xg']:.2f}"
            )
            with st.expander("Probability evolution and model details"):
                minutes = sorted(set([0, selected_minute] + list(range(15, selected_minute + 1, 15))))
                points = []
                for m in minutes:
                    gh, ga = score_at(observed, m, home_id, away_id)
                    p = predict(history_matches, fixture, m, gh, ga)
                    points.append((m, p))
                fig = go.Figure()
                for key, label, color in [
                    ("home_win", home_name, HOME_COLOR),
                    ("draw", "Draw", "#d6a23c"),
                    ("away_win", away_name, AWAY_COLOR),
                ]:
                    fig.add_trace(
                        go.Scatter(
                            x=[m for m, _ in points],
                            y=[100 * p[key] for _, p in points],
                            name=label,
                            mode="lines+markers",
                            line=dict(color=color),
                        )
                    )
                fig.update_layout(
                    height=300,
                    margin=dict(l=10, r=10, t=10, b=10),
                    xaxis=dict(title="Match minute", range=[0, 90]),
                    yaxis=dict(title="Probability (%)", range=[0, 100]),
                )
                st.plotly_chart(fig, use_container_width=True)
                st.caption(
                    "Goal rates estimated from historical matches strictly before kickoff; "
                    "home/away attack and defence rates are smoothed toward league averages. "
                    "Remaining goals follow independent Poisson distributions. "
                    "The model responds to score and time, but not yet xG, cards or substitutions. "
                    "Synthetic demonstration; probabilities are not externally calibrated."
                )
    except (ValueError, KeyError, TypeError, OSError) as exc:
        st.warning(f"Outcome prediction unavailable: {exc}")

tab_names = [
    "📊 Match statistics",
    "👥 Lineups",
    "👤 Player performance",
    "✨ AI analysis",
    "🏆 League standings and statistics",
    "📈 Form",
]
if phase in ("halftime", "live"):
    tab_names.insert(4, "🔁 Substitution candidates")
tabs = st.tabs(tab_names)
stats_tab, lineup_tab, player_tab, analysis_tab = tabs[:4]
if phase in ("halftime", "live"):
    tactics_tab, league_tab, form_tab = tabs[4:]
else:
    tactics_tab, league_tab, form_tab = None, tabs[4], tabs[5]
with lineup_tab:
    show_lineups(replay_payload, home_name, away_name, home_id, away_id, player_names, player_positions)
with player_tab:
    show_player_performance(payload, player_names, player_positions, home_name, away_name)
with stats_tab:
    if phase == "prematch":
        st.info(
            "Match statistics become available after kickoff. Historical information is in the Form and League standings and statistics tabs."
        )
    else:
        home_shots = replay_state.get("shots", {}).get(home_id, {})
        away_shots = replay_state.get("shots", {}).get(away_id, {})
        st.markdown("#### Attacking")
        for label, key, decimals in [
            ("Shots", "shots", 0),
            ("Shots on target", "shots_on_target", 0),
            ("Expected goals (xG)", "xg", 2),
        ]:
            compare_bar(label, home_shots.get(key, 0), away_shots.get(key, 0), home_name, away_name, decimals)
        home_half = replay_state.get("completed_half_stats", {}).get(home_id, {})
        away_half = replay_state.get("completed_half_stats", {}).get(away_id, {})
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
        events = pd.DataFrame(replay_payload.get("observed_events", []))
        if not events.empty and {"minute", "event_type", "team_id"}.issubset(events.columns):
            st.markdown("#### Shot timeline")
            shots = events[events.event_type.isin(["shot", "goal"])].copy()
            if not shots.empty:
                shots["team"] = shots.team_id.astype(str).map({home_id: home_name, away_id: away_name})
                shots["xg"] = pd.to_numeric(shots.get("xg", 0), errors="coerce").fillna(0)
                fig = go.Figure()
                for team, color in [(home_name, HOME_COLOR), (away_name, AWAY_COLOR)]:
                    subset = (
                        shots[shots.team == team]
                        .sort_values(["minute", "second"] if "second" in shots.columns else ["minute"])
                        .copy()
                    )
                    if subset.empty:
                        continue
                    subset["cumulative_xg"] = subset.xg.cumsum()
                    # Goals retain their true position on the cumulative-xG curve.
                    fig.add_trace(
                        go.Scatter(
                            x=subset.minute,
                            y=subset.cumulative_xg,
                            mode="lines+markers",
                            name=team,
                            line=dict(color=color),
                            marker=dict(size=7),
                            customdata=subset.event_type,
                            hovertemplate="%{x}′ · Cumulative xG %{y:.2f}<br>%{customdata}<extra></extra>",
                        )
                    )
                    goals = subset[subset.event_type == "goal"]
                    if not goals.empty:
                        fig.add_trace(
                            go.Scatter(
                                x=goals.minute,
                                y=goals.cumulative_xg,
                                mode="markers",
                                name=f"{team} goals",
                                marker=dict(size=17, color=color, symbol="star", line=dict(color="white", width=2)),
                                customdata=[
                                    (
                                        player_names.get(str(pid), "Scorer unavailable")
                                        if pd.notna(pid)
                                        else "Scorer unavailable"
                                    )
                                    for pid in goals.get("player_id", pd.Series([None] * len(goals), index=goals.index))
                                ],
                                hovertemplate="GOAL · %{x}′<br>%{customdata}<br>Cumulative xG %{y:.2f}<extra></extra>",
                            )
                        )
                fig.update_layout(
                    height=340, margin=dict(l=20, r=20, t=20, b=20), xaxis_title="Minute", yaxis_title="Cumulative xG"
                )
                st.plotly_chart(fig, use_container_width=True)
            # Bookings are timestamped events, not estimates from half totals.
            cards = events[events.event_type.isin(["yellow_card", "red_card", "second_yellow_red"])].copy()
            if not cards.empty:
                st.markdown("#### Discipline")
                c_home, c_away = st.columns(2)
                for col, tid, name in [(c_home, home_id, home_name), (c_away, away_id, away_name)]:
                    club_cards = cards[cards.team_id.astype(str) == str(tid)]
                    yellow = int((club_cards.event_type == "yellow_card").sum())
                    red = int(club_cards.event_type.isin(["red_card", "second_yellow_red"]).sum())
                    col.markdown(f"**{name}** · 🟨 {yellow} · 🟥 {red}")
            st.markdown("#### Match events")
            # Resolve actual event participants; unattributed historical shots stay unknown.
            event_table = events.copy()
            if "player_id" in event_table.columns:
                event_table["Player"] = event_table["player_id"].apply(
                    lambda pid: player_names.get(str(pid), "Unknown player") if pd.notna(pid) else "Not attributed"
                )
            else:
                event_table["Player"] = "Not attributed"
            if "team_id" in event_table.columns:
                event_table["Team"] = (
                    event_table["team_id"].astype(str).map({home_id: home_name, away_id: away_name}).fillna("Unknown")
                )
            else:
                event_table["Team"] = "Unknown"
            event_table["Event"] = event_table["event_type"].astype(str).str.replace("_", " ").str.title()
            display = [
                c for c in ("minute", "Team", "Event", "Player", "description", "xg") if c in event_table.columns
            ]
            event_table = event_table[display].rename(
                columns={"minute": "Minute", "description": "Description", "xg": "xG"}
            )
            st.dataframe(event_table.sort_values("Minute", ascending=False), hide_index=True, use_container_width=True)

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


with league_tab:
    show_league_position_impact(payload)
    show_league(payload, phase, home_id, away_id)
    st.divider()
    show_league_player_leaders(payload, phase)

with form_tab:
    if phase == "live":
        st.markdown("#### Historical performance by match interval")
        show_interval_comparison(payload, home_id, away_id, home_name, away_name)
    show_history(payload.get("historical_context", {}), home_name, away_name)

st.divider()
st.caption(
    "All clubs, players and matches are fictional. Data is generated locally; narratives are created separately via Azure AI."
)
