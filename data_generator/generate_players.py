"""Generate fictional squads and hidden player ability profiles.

Prerequisite: run generate_teams.py in the same directory.
Install: pip install numpy pandas pyarrow
Run: python generate_players.py

Outputs:
  data/players.parquet         player identities and stable descriptors
  data/player_profiles.parquet hidden simulation abilities (NOT observed per-90 stats)
"""

from pathlib import Path
import numpy as np
import pandas as pd

SEED = 43  # Independent of the team generator's random stream.
DATA_DIR = Path(__file__).resolve().parent / "data"

# Configurable squad distribution: 22 players per team.
SQUAD_COMPOSITION = {"GK": 2, "CB": 4, "WB": 4, "CM": 4, "AMW": 5, "ST": 3}
ABILITY_COLUMNS = ["attacking", "passing", "defensive", "possession", "physical"]

# Baseline underlying abilities in the order of ABILITY_COLUMNS.
POSITION_BASELINES = {
    "GK": [15, 45, 60, 35, 60],
    "CB": [25, 55, 77, 50, 72],
    "WB": [47, 62, 61, 65, 76],
    "CM": [54, 73, 63, 73, 67],
    "AMW": [76, 67, 34, 76, 70],
    "ST": [79, 48, 34, 62, 72],
}

# Archetype modifiers encourage recognizable specializations and trade-offs.
ARCHETYPES = {
    "GK": {
        "shot_stopper": [0, -7, 12, -4, 2],
        "sweeper_keeper": [0, 12, -4, 9, 5],
    },
    "CB": {
        "ball_playing": [-2, 14, -5, 10, -3],
        "stopper": [0, -8, 13, -7, 7],
        "aerial_specialist": [2, -7, 9, -4, 12],
    },
    "WB": {
        "attacking_fullback": [11, 8, -10, 11, 3],
        "defensive_fullback": [-9, -2, 13, -5, 4],
        "inverted_fullback": [-1, 14, 0, 10, -4],
    },
    "CM": {
        "deep_playmaker": [-4, 15, -2, 10, -7],
        "ball_winner": [-10, -8, 16, -6, 9],
        "box_to_box": [6, -3, 5, 3, 12],
        "progressive_midfielder": [6, 10, -8, 13, -3],
    },
    "AMW": {
        "creator": [-2, 15, -5, 11, -6],
        "dribbler": [4, -6, -6, 17, 7],
        "inside_forward": [13, -6, -4, 6, 5],
        "pressing_attacker": [-2, -4, 11, -4, 12],
    },
    "ST": {
        "poacher": [15, -9, -6, 1, -3],
        "target_forward": [5, -7, 2, -8, 15],
        "pressing_forward": [-4, 0, 10, 2, 13],
        "complete_forward": [7, 10, 0, 7, 2],
    },
}

FIRST_NAMES = [
    "Adam",
    "Alex",
    "Andrej",
    "Anton",
    "Benjamin",
    "Bruno",
    "Carlos",
    "Daniel",
    "David",
    "Diego",
    "Dominik",
    "Elias",
    "Emil",
    "Erik",
    "Felix",
    "Filip",
    "Gabriel",
    "Hugo",
    "Ivan",
    "Jakub",
    "Jan",
    "Jonas",
    "Julian",
    "Kevin",
    "Lars",
    "Leo",
    "Liam",
    "Lucas",
    "Luka",
    "Marco",
    "Martin",
    "Mateo",
    "Max",
    "Milan",
    "Milos",
    "Nico",
    "Noah",
    "Oliver",
    "Oscar",
    "Pablo",
    "Patrick",
    "Paul",
    "Petr",
    "Rafael",
    "Roman",
    "Samuel",
    "Simon",
    "Stefan",
    "Theo",
    "Thomas",
    "Tomas",
    "Victor",
    "Viktor",
    "William",
    "Yann",
    "Zoran",
]
LAST_NAMES = [
    "Andersen",
    "Bauer",
    "Bennett",
    "Costa",
    "Davis",
    "Diaz",
    "Dvorak",
    "Eriksen",
    "Fischer",
    "Garcia",
    "Gomez",
    "Hansen",
    "Horak",
    "Ivanov",
    "Jensen",
    "Keller",
    "Kolar",
    "Kovac",
    "Kral",
    "Larsen",
    "Lopez",
    "Marin",
    "Mendez",
    "Meyer",
    "Miller",
    "Morales",
    "Muller",
    "Navarro",
    "Nielsen",
    "Novak",
    "Oliveira",
    "Ortega",
    "Pavlov",
    "Pereira",
    "Petrov",
    "Popov",
    "Ramos",
    "Richter",
    "Rivera",
    "Rossi",
    "Santos",
    "Schmidt",
    "Silva",
    "Sokolov",
    "Sosa",
    "Stein",
    "Svoboda",
    "Torres",
    "Urban",
    "Varga",
    "Vasquez",
    "Vega",
    "Vidal",
    "Weber",
    "Wolf",
    "Zeman",
]

SECONDARY_POSITIONS = {
    "GK": [None],
    "CB": [None, "WB", "CM"],
    "WB": [None, "CB", "AMW"],
    "CM": [None, "CB", "AMW"],
    "AMW": [None, "CM", "ST", "WB"],
    "ST": [None, "AMW"],
}
HEIGHT_PARAMS = {"GK": (190, 5), "CB": (188, 6), "WB": (177, 6), "CM": (180, 6), "AMW": (176, 6), "ST": (184, 7)}


def generate_players(
    teams: pd.DataFrame,
    team_profiles: pd.DataFrame,
    seed: int = SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return public players and hidden player ability profiles."""
    rng = np.random.default_rng(seed)
    if teams["team_id"].duplicated().any() or team_profiles["team_id"].duplicated().any():
        raise ValueError("Team IDs must be unique")
    merged = teams[["team_id"]].merge(team_profiles, on="team_id", how="left", validate="one_to_one")
    if merged["overall_strength"].isna().any():
        raise ValueError("Some teams have no profile")

    player_rows, profile_rows = [], []
    used_names = set()
    player_index = 0

    for team in merged.itertuples(index=False):
        for position, count in SQUAD_COMPOSITION.items():
            archetype_names = list(ARCHETYPES[position])
            # Spread archetypes across the squad instead of pure random sampling.
            assigned = (archetype_names * ((count // len(archetype_names)) + 1))[:count]
            rng.shuffle(assigned)
            for slot, archetype in enumerate(assigned):
                player_index += 1
                player_id = f"P{player_index:04d}"
                for _ in range(100):
                    name = f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"
                    if name not in used_names:
                        break
                else:
                    raise RuntimeError("Unable to generate a unique player name")
                used_names.add(name)

                # Stronger clubs have better players on average, with overlap and
                # occasional stars at weaker clubs / weaker backups at strong clubs.
                team_effect = (float(team.overall_strength) - 65) * 0.42
                player_quality = rng.normal(0, 8.5)
                # Slot-based depth: modestly weaker later players, not a rigid ranking.
                depth_effect = -2.0 * slot
                baseline = np.array(POSITION_BASELINES[position], dtype=float)
                archetype_effect = np.array(ARCHETYPES[position][archetype], dtype=float)
                shared_quality = team_effect + player_quality + depth_effect
                # Correlated skill categories via shared quality, plus distinct noise.
                ability = np.clip(
                    np.rint(baseline + archetype_effect + shared_quality + rng.normal(0, 6, 5)), 10, 98
                ).astype(int)

                height_mean, height_sd = HEIGHT_PARAMS[position]
                age = int(np.clip(np.rint(rng.normal(26, 4.2)), 18, 37))
                foot = str(rng.choice(["right", "left", "both"], p=[0.70, 0.25, 0.05]))
                secondary = rng.choice(
                    SECONDARY_POSITIONS[position],
                    p=(
                        [1.0]
                        if position == "GK"
                        else [0.65]
                        + [0.35 / (len(SECONDARY_POSITIONS[position]) - 1)] * (len(SECONDARY_POSITIONS[position]) - 1)
                    ),
                )
                player_rows.append(
                    {
                        "player_id": player_id,
                        "team_id": team.team_id,
                        "player_name": name,
                        "age": age,
                        "primary_position": position,
                        "secondary_position": secondary,
                        "archetype": archetype,
                        "preferred_foot": foot,
                        "height_cm": int(np.clip(np.rint(rng.normal(height_mean, height_sd)), 160, 205)),
                    }
                )
                profile_rows.append(
                    {
                        "player_id": player_id,
                        "team_id": team.team_id,
                        "overall_ability": int(round(float(np.mean(ability)))),
                        **{f"{col}_ability": int(ability[i]) for i, col in enumerate(ABILITY_COLUMNS)},
                    }
                )

    players = pd.DataFrame(player_rows)
    profiles = pd.DataFrame(profile_rows)
    validate_players(players, profiles, teams)
    return players, profiles


def validate_players(players: pd.DataFrame, profiles: pd.DataFrame, teams: pd.DataFrame) -> None:
    assert players["player_id"].is_unique
    assert profiles["player_id"].is_unique
    assert set(players["player_id"]) == set(profiles["player_id"])
    assert set(players["team_id"]) == set(teams["team_id"])
    assert (players.groupby("team_id").size() == sum(SQUAD_COMPOSITION.values())).all()
    counts = players.groupby(["team_id", "primary_position"]).size().unstack(fill_value=0)
    for position, expected in SQUAD_COMPOSITION.items():
        assert (counts[position] == expected).all(), f"Unexpected squad count for {position}"
    for col in ["overall_ability", *[f"{x}_ability" for x in ABILITY_COLUMNS]]:
        assert profiles[col].between(0, 100).all(), f"Invalid {col}"
    assert players["age"].between(18, 37).all()
    assert players["player_name"].is_unique


def main() -> None:
    teams_path = DATA_DIR / "teams.parquet"
    profiles_path = DATA_DIR / "team_profiles.parquet"
    if not teams_path.exists() or not profiles_path.exists():
        raise FileNotFoundError(
            "Run generate_teams.py first to create data/teams.parquet and data/team_profiles.parquet"
        )
    teams = pd.read_parquet(teams_path)
    team_profiles = pd.read_parquet(profiles_path)
    players, profiles = generate_players(teams, team_profiles)
    players.to_parquet(DATA_DIR / "players.parquet", index=False)
    profiles.to_parquet(DATA_DIR / "player_profiles.parquet", index=False)
    print(f"Generated {len(players)} players across {len(teams)} teams (seed={SEED})")
    print("\nPosition distribution:\n", players["primary_position"].value_counts().to_string())
    print("\nExample squad:\n", players[players.team_id == teams.iloc[0].team_id].to_string(index=False))
    print(f"\nFiles saved to: {DATA_DIR}")


if __name__ == "__main__":
    main()
