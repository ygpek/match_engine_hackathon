# ⚽ Match Studio AI

**A phase-aware football analytics and AI match-reporting prototype**

Match Studio AI is a hackathon project demonstrating how structured football data, statistical analysis, tactical decision support, and generative AI can work together in a single interactive match-analysis workflow.

The application follows a fictional league fixture through **pre-match, half-time, live, and post-match** stages. It combines synthetic historical datasets with deterministic analytics engines, prepares phase-specific evidence, generates grounded reports through **Azure OpenAI**, and presents the result in a **Streamlit** dashboard.

> **Demo disclaimer:** Every team, player, fixture and performance observation is synthetic. Rankings and tactical suitability scores are descriptive heuristics, not validated forecasts or real-world scouting assessments. The hosted demo can run from saved outputs without calling Azure.

## 1. End-to-end architecture

The architecture intentionally separates **data generation**, **quantitative analysis**, **evidence assembly**, **AI narration**, and **presentation**. The language model explains structured evidence; it does not generate the underlying match facts or compute the analytics.

## 2. Phase I — Synthetic football data generation

The project starts without a paid football-data feed. Instead, `data_generator/` creates a coherent fictional competition with **20 clubs, approximately 440 players, and two historical league seasons (760 fixtures)**.

Run the historical generation pipeline in dependency order:

```bash
python data_generator/generate_data.py
```

The script invokes the following generators:

| Order | Generator | Purpose |
|---|---|---|
| 1 | `generate_teams.py` | Creates fictional clubs and competitive-tier/team profiles. |
| 2 | `generate_players.py` | Creates player rosters, positions and player profiles. |
| 3 | `generate_historical_matches.py` | Produces historical schedules, fixtures, scores, lineups and shot-level match observations. |
| 4 | `generate_historical_team_stats.py` | Produces team-level match and half-level statistical aggregates. |
| 5 | `generate_historical_player_match_stats.py` | Builds player-match observations, including minutes, position-relevant contributions and disciplinary data. |
| 6 | `generate_historical_player_season_stats.py` | Aggregates player observations into season and career-level summaries. |

### Data model

Generated datasets are stored as **Parquet** in `data_generator/data/`. The key relationships are:

- `team_id` connects teams, fixtures, team statistics, and players.
- `player_id` connects rosters, lineups, shots, player-match records, and player profiles.
- `match_id` connects fixtures, lineups, shot events, disciplinary events, and match-level statistics.
- `season_id` identifies the season for historical filtering and league tables.

Representative outputs include:

| Dataset | Contents |
|---|---|
| `teams.parquet`, `team_profiles.parquet` | Club identities and competitive tiers. |
| `players.parquet`, `player_profiles.parquet` | Player identities, positions and roster attributes. |
| `historical_matches.parquet` | Fixture dates, participants and results. |
| `historical_lineups.parquet` | Match participants, starts and substitution timing. |
| `historical_shots.parquet` | Shot events, xG and goal outcomes, with player attribution where available. |
| `historical_disciplinary_events.parquet` | Timestamped cautions and dismissals. |
| `historical_team_stats.parquet`, `historical_team_shot_stats.parquet` | Team-level match and shot statistics. |
| `historical_player_match_stats.parquet` | Individual match performances. |
| `historical_player_season_stats.parquet`, `historical_player_career_stats.parquet` | Aggregated historical performance. |
| `player_similarity_profiles.parquet`, `player_similarity_top10.parquet` | Precomputed similarity-engine artifacts. |

The synthetic event layer is designed to be internally consistent: goals are tied to shot outcomes, players are checked against the fixture's teams and lineups, and match snapshots are reconstructed from the same underlying events. Team statistics that exist only as half-level totals are **not** assigned invented minute-by-minute timestamps.

## 3. Phase II — Statistical and tactical analytics engines

The `engines/` directory contains three core analytical components. They answer different questions and should not be treated as interchangeable.

### A. Player similarity — `engines/player_similarity_engine.py`

**Question:** *Which players have comparable statistical profiles?*

The engine uses observed season statistics rather than hidden player-ability ratings. It:

1. Builds player profiles from historical per-90 and percentage metrics.
2. Applies playing-time eligibility thresholds and a recent-season weighting.
3. Organizes features into **attacking, passing, defending, possession and physical** groups.
4. Standardizes the comparison features.
5. Uses **Principal Component Analysis (PCA)** to retain approximately 90% of feature variance.
6. Computes **cosine similarity** in the PCA space and ranks comparable players, typically within the same position.

The standalone engine writes `player_similarity_profiles.parquet` and `player_similarity_top10.parquet`. The Streamlit interface also has a **phase-safe recent-appearances similarity implementation** in `match_studio/player_similarity_ui.py`, using only matches before the featured fixture. It supports a player-comparison radar based on same-position percentile ranks.

**Interpretation:** similarity describes statistical resemblance, not player quality, transfer value, or future performance. Dedicated goalkeeper similarity is not implemented because the generated data lacks sufficient shot-stopping features.

### B. Tactical suitability — `engines/tactical_suitability_engine.py`

**Question:** *Which eligible player best matches a specific tactical objective?*

This engine uses position-relative statistical percentiles and weighted feature groups to rank players for six scenarios:

- Balanced
- Chase a goal
- Protect the lead
- Control midfield
- High press
- Counterattack

For example, a *chase a goal* scenario places more weight on attacking and possession-related evidence, while *protect the lead* emphasizes defending, passing and possession. The engine compares candidates against a same-position reference pool, then filters to eligible players.

The interactive Streamlit simulator uses `match_studio/tactical_simulator.py` to compare a selected on-field player with a compatible bench candidate. Its score difference is a **relative suitability comparison**, not a predicted increase in win probability, goals, or match outcome.

### C. Historical context — `engines/historical_context_engine.py`

**Question:** *What does the team's previous match history tell us about the current situation?*

The engine builds evidence from earlier fixtures, including:

- Recent wins, draws, losses and points.
- Home/away form and opponent form.
- Historical head-to-head records where available.
- Outcomes from similar half-time score situations.
- Team-level scoring, xG, shot and possession patterns.

A strict date cutoff prevents the featured match or future fixtures from entering the historical evidence. These results provide context and sample sizes, not causal explanations or guaranteed predictions.

## 4. Phase III — Featured match selection and reconstruction

`featured_match/generate_featured_match.py` selects a fixture from the generated historical competition and converts it into the match followed by the application.

```bash
python featured_match/generate_featured_match.py
```

The generator supports **random selection** (the default), a **dramatic match pattern**, an explicit match ID, a random seed, and a configurable live snapshot minute (default: **67′**). It reconstructs a consistent event timeline from the selected historical fixture instead of independently inventing new results for each phase.

The resulting `featured_match/data/` directory contains:

```text
featured_match.json
featured_match_snapshots.json
featured_match_lineups.parquet
featured_match_events.parquet
featured_match_shots.parquet
featured_match_team_half_stats.parquet
featured_match_player_stats.parquet
```

The generator validates the featured players against their teams and lineups and checks goal attribution consistency before writing outputs.

### Four match phases

| Phase | Available match evidence | What remains hidden |
|---|---|---|
| **Pre-match** | Fixture identity, available squad information and pre-match history. | Goals, bookings and all future match events. |
| **Half-time** | Observed first-half events and completed first-half aggregates, through 45′. | All second-half events and the final result. |
| **Live** | Observed events through the configured live minute (67′ by default), current score and eligible players. | Events after the live cutoff and the final result. |
| **Post-match** | Full fixture events, final score and complete match evidence. | Nothing from this featured fixture. |

**Phase safety is a core design principle:** every snapshot is constructed at the appropriate cutoff. A user viewing the live phase must not accidentally see a future goal, red card or substitution.

## 5. Phase IV — Match Studio orchestration

`match_studio/match_studio_orchestrator.py` integrates the featured match with the analytical engines and produces a structured evidence package for each phase.

```bash
python match_studio/match_studio_orchestrator.py
```

For each phase, the orchestrator:

1. Loads the featured fixture, event timeline, lineups and phase snapshot.
2. Restricts historical data to matches **strictly before kickoff**, excluding the featured fixture.
3. Builds historical team context and player profiles using only eligible data.
4. Determines current on-pitch players and available substitution candidates.
5. Generates tactical options and same-position substitution comparisons.
6. Assembles score, xG, observed events, goal attribution and disciplinary summaries.
7. Adds individual player-performance review and league-expectation evidence where appropriate.
8. Serializes the resulting evidence into phase-specific JSON.

Outputs:

```text
match_studio/output/match_studio_prematch.json
match_studio/output/match_studio_halftime.json
match_studio/output/match_studio_live.json
match_studio/output/match_studio_postmatch.json
```

These JSON files are the **source of truth passed to the narrative layer**. Keeping evidence generation separate from language generation makes it easier to audit numerical claims and investigate inconsistencies.

## 6. Phase V — Grounded AI reporting with Azure OpenAI

`match_studio/azure_narrative_engine.py` converts the orchestrator's structured evidence into readable football analysis. The project uses an **Azure OpenAI deployment of GPT-4.1 mini**.

The model is prompted to act as a football analyst, but with explicit restrictions:

- Use only facts supplied in the phase's evidence JSON.
- Never reveal future match events in pre-match, half-time or live reports.
- Do not invent goalscorers, assists, bookings, injuries or tactical actions.
- Attribute featured-match goals only when the authoritative goal summary identifies a scorer.
- Recommend only players who are actually eligible in the relevant phase.
- Treat historical form and tactical scores as descriptive evidence, not outcome probabilities.
- Cite significant numerical claims with evidence paths in structured metadata.
- Keep internal JSON keys, player IDs and algorithm terminology out of the public-facing prose.

The engine checks the report structure and validates referenced evidence paths. It can request a correction when a generated response fails validation. The reports are written as **JSON** for the application and **Markdown** for human review:

```text
match_studio/narratives/match_studio_prematch_report.json
match_studio/narratives/match_studio_halftime_report.json
match_studio/narratives/match_studio_live_report.json
match_studio/narratives/match_studio_postmatch_report.json
```

To regenerate reports with Azure, set the following variables in a local `.env` file (never commit the file):

```dotenv
AZURE_OPENAI_ENDPOINT=...
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_DEPLOYMENT=...
```

Then run:

```bash
python match_studio/azure_narrative_engine.py --dry-run
python match_studio/azure_narrative_engine.py
```

`--dry-run` previews prompts without contacting Azure. The default execution generates the phase reports. **The public demo does not need these credentials**, because it reads the previously saved narratives.

## 7. Phase VI — Streamlit application

The final layer, `match_studio/app.py`, brings the evidence, analytics and AI commentary into one interactive interface.

```bash
streamlit run match_studio/app.py
```

### Main features

| Interface | Capabilities |
|---|---|
| **Match statistics** | Score, goalscorers, shots, shots on target, cumulative xG, goal markers, bookings and event table. |
| **Interactive replay** | Minute-by-minute exploration of recorded events and score, restricted to the selected phase. |
| **Lineups** | Illustrative pitch layouts, active players, substitutions and dismissals. |
| **Player performance** | Historical per-90 profiles, observed participation, similar players from other clubs and a position-specific radar comparison. |
| **AI analysis** | Pre-generated, phase-specific Azure OpenAI narrative reports. |
| **Substitution candidates** | Interactive tactical scenario selection and same-position player suitability comparisons in half-time/live phases. |
| **League standings and statistics** | Season standings, highlighted featured clubs, postmatch position impact, and individual league leaders. |
| **Form** | Recent form, head-to-head context and historical 15-minute interval comparisons. |

The replay control follows the match's information boundaries:

| Phase | Replay slider |
|---|---|
| Pre-match | Not available |
| Half-time | 0–45′ |
| Live | 0–configured live minute |
| Post-match | 0–90′ |

Moving the slider updates the **observed match presentation**, not the official phase's AI report, historical evidence, or tactical recommendation set. The interface therefore remains interactive without generating a new model response for every slider movement.

### Deployment model

The hosted version is deliberately **read-only**. It loads committed Parquet and JSON artifacts and makes no Azure API calls during normal browsing. This simplifies deployment, keeps demo costs predictable and ensures all visitors explore the same reproducible fictional fixture.

## 8. Repository structure

```text
.
├── data_generator/
│   ├── generate_data.py
│   ├── generate_*.py
│   └── data/                        # Synthetic historical Parquet datasets
├── engines/
│   ├── player_similarity_engine.py
│   ├── tactical_suitability_engine.py
│   └── historical_context_engine.py
├── featured_match/
│   ├── generate_featured_match.py
│   └── data/                        # Featured fixture, snapshots and events
├── match_studio/
│   ├── app.py                       # Streamlit entrypoint
│   ├── match_studio_orchestrator.py
│   ├── azure_narrative_engine.py
│   ├── historical_visuals.py
│   ├── league_expectations.py
│   ├── league_player_leaders.py
│   ├── player_performance.py
│   ├── player_performance_engine.py
│   ├── player_similarity_ui.py
│   ├── tactical_simulator.py
│   ├── output/                      # Structured evidence per phase
│   └── narratives/                  # Generated AI reports
├── requirements.txt
└── README.md
```

## 9. Run locally

### View the existing demo

With Python installed, create a virtual environment and install dependencies:

```bash
python -m venv .venv
```

Activate it with `.venv\Scripts\activate` on Windows or `source .venv/bin/activate` on macOS/Linux, then:

```bash
pip install -r requirements.txt
streamlit run match_studio/app.py
```

The repository should contain the generated `data_generator/data/`, `featured_match/data/`, `match_studio/output/`, and `match_studio/narratives/` artifacts for this read-only mode.

### Rebuild the synthetic scenario (optional)

If you want to generate a different fictional match and new reports, run the pipeline in order:

```bash
python data_generator/generate_data.py
python engines/player_similarity_engine.py
python featured_match/generate_featured_match.py
python match_studio/match_studio_orchestrator.py
python match_studio/azure_narrative_engine.py
streamlit run match_studio/app.py
```

The tactical suitability and historical context engines are also available as standalone command-line tools; the orchestrator uses their logic to assemble phase-specific recommendations and context. Regenerating the data changes the demo fixture and can overwrite previously saved artifacts. Running the Azure narrative engine requires valid Azure credentials and may incur API charges.

## 10. Technology stack

- **Python** — data generation, transformation, orchestration and analytics.
- **pandas / NumPy / PyArrow** — tabular processing and Parquet storage.
- **scikit-learn** — feature scaling, PCA and cosine similarity.
- **Plotly** — interactive xG, pitch, interval and radar visualizations.
- **Streamlit** — interactive web application.
- **Azure OpenAI (GPT-4.1 mini deployment)** — evidence-grounded football narratives.
- **GitHub + Streamlit Community Cloud** — source hosting and demo deployment.

## 11. Limitations and next steps

This is a proof of concept, not a production scouting or match-tracking system. Its most important limitations are:

- All football observations are synthetic; they do not represent real clubs or players.
- Tactical suitability is a statistical heuristic, not a causal simulation of substitutions.
- Pitch positions are illustrative, not derived from tracking data.
- Half-level team aggregates cannot be faithfully replayed minute by minute.
- Player similarity and radar percentiles describe statistical profiles, not absolute talent.
- Goalkeeper comparisons need richer save and shot-stopping data.
- AI reports are validated against structured evidence, but their football interpretations still require human judgment.

A production version would integrate licensed event/tracking data, stronger validation and monitoring, a persistent database, and live ingestion before supporting operational coaching decisions.

---

**Match Studio AI** demonstrates a complete analytical workflow: **synthetic data → historical evidence → statistical engines → phase-safe match intelligence → grounded AI reporting → interactive Streamlit product**.
