"""Generate reproducible fictional football teams and latent team profiles.

Install: pip install numpy pandas pyarrow
Run:     python generate_teams.py

Competitive tier determines quality distribution, NOT tactical style.
Profiles are hidden simulator inputs, not observed match statistics.
"""

from pathlib import Path
import numpy as np
import pandas as pd

SEED = 55
OUTPUT_DIR = Path(__file__).resolve().parent / "data"

TEAM_NAMES = [
    "North City",
    "Capital United",
    "Riverside FC",
    "Harbor Athletic",
    "Eastborough",
    "Westford Town",
    "Kingsport FC",
    "Valley Rovers",
    "Ironbridge United",
    "Lakeside FC",
    "Summit Athletic",
    "Oldcastle FC",
    "Meadowfield",
    "Redmont United",
    "Southgate FC",
    "Pinehurst City",
    "Coastal Rovers",
    "Stonehaven FC",
    "Brookfield Town",
    "Highland Athletic",
]

# (count, average underlying quality, standard deviation)
TIER_CONFIG = {
    "title_contender": (3, 87, 3.5),
    "european_contender": (5, 76, 4.0),
    "mid_table": (8, 64, 5.0),
    "relegation_battle": (4, 51, 4.5),
}

# Means for possession_style, pressing_intensity, directness, tempo.
# These are tendencies, not success/quality ratings.
STYLE_PROFILES = {
    "possession": (85, 68, 25, 57),
    "high_press": (64, 88, 53, 78),
    "counterattack": (38, 53, 86, 77),
    "direct_physical": (34, 58, 84, 62),
    "balanced": (55, 55, 52, 55),
}
STYLE_WEIGHTS = {
    "possession": 0.22,
    "high_press": 0.20,
    "counterattack": 0.22,
    "direct_physical": 0.16,
    "balanced": 0.20,
}
STYLE_COLUMNS = ["possession_style", "pressing_intensity", "directness", "tempo"]


def generate_teams(seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    n = len(TEAM_NAMES)
    if sum(config[0] for config in TIER_CONFIG.values()) != n:
        raise ValueError("Tier counts must equal the number of teams")

    teams = pd.DataFrame(
        {
            "team_id": [f"T{i:03d}" for i in range(1, n + 1)],
            "team_name": TEAM_NAMES,
            "short_name": [name.split()[0][:3].upper() for name in TEAM_NAMES],
            "country": "Synthetic Republic",
            "league_id": "LEAGUE_01",
        }
    )

    # Generate exact tier counts, then shuffle assignments independently of names.
    tiers = [tier for tier, (count, _, _) in TIER_CONFIG.items() for _ in range(count)]
    rng.shuffle(tiers)
    base_quality = np.array([rng.normal(TIER_CONFIG[tier][1], TIER_CONFIG[tier][2]) for tier in tiers])
    # Attack and defense share a quality factor but can differ substantially.
    attack = np.clip(np.rint(base_quality + rng.normal(0, 5.5, n)), 30, 95).astype(int)
    defense = np.clip(np.rint(base_quality + rng.normal(0, 5.5, n)), 30, 95).astype(int)
    overall = np.rint((attack + defense) / 2).astype(int)

    # Style is sampled independently of competitive tier and team quality.
    styles = rng.choice(list(STYLE_WEIGHTS), size=n, p=list(STYLE_WEIGHTS.values()))
    style_baselines = np.array([STYLE_PROFILES[style] for style in styles])
    style_values = np.clip(np.rint(style_baselines + rng.normal(0, 8, size=(n, len(STYLE_COLUMNS)))), 15, 95).astype(
        int
    )

    profiles = pd.DataFrame(
        {
            "team_id": teams["team_id"],
            "competitive_tier": tiers,
            "tactical_style": styles,
            "overall_strength": overall,
            "attack_strength": attack,
            "defense_strength": defense,
            **{col: style_values[:, i] for i, col in enumerate(STYLE_COLUMNS)},
        }
    )
    validate_teams(teams, profiles)
    return teams, profiles


def validate_teams(teams: pd.DataFrame, profiles: pd.DataFrame) -> None:
    assert teams["team_id"].is_unique
    assert profiles["team_id"].is_unique
    assert set(teams["team_id"]) == set(profiles["team_id"])
    assert profiles["competitive_tier"].value_counts().to_dict() == {
        tier: config[0] for tier, config in TIER_CONFIG.items()
    }
    for col in ["overall_strength", "attack_strength", "defense_strength", *STYLE_COLUMNS]:
        assert profiles[col].between(0, 100).all(), f"Out-of-range values in {col}"


def main() -> None:
    teams, profiles = generate_teams()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    teams.to_parquet(OUTPUT_DIR / "teams.parquet", index=False)
    profiles.to_parquet(OUTPUT_DIR / "team_profiles.parquet", index=False)
    print(f"Generated {len(teams)} teams (seed={SEED})")
    print(teams.merge(profiles, on="team_id").to_string(index=False))
    print("\nCompetitive tier counts:\n", profiles["competitive_tier"].value_counts().to_string())
    print(f"\nFiles saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
