from dataclasses import dataclass


@dataclass
class Config:
    """Every number of the game lives here. Change numbers here, nowhere else."""

    # --- Levels (feature switches) ---
    level: int = 4  # 1 = dodge only ... 4 = everything

    # --- Arena ---
    arena_size: int = 600

    # --- Ship ---
    ship_radius: float = 15.0
    ship_speed: float = 6.0
    ship_start_offset: float = 60.0
    start_lives: int = 3
    invulnerable_steps: int = 60

    # --- Shooting and ammo ---
    bullet_radius: float = 4.0
    bullet_speed: float = 12.0
    shoot_cooldown: int = 8
    max_ammo: int = 20
    ammo_recharge_steps: int = 20

    # --- Enemies ---
    enemy_radius: float = 15.0
    drifter_speed: float = 3.0
    hunter_speed: float = 1.5
    hunter_probability: float = 0.25
    hunter_hp: int = 2
    spawn_start_interval: int = 40
    spawn_min_interval: int = 15

    # --- Shield pickup ---
    pickup_radius: float = 10.0
    pickup_speed: float = 2.0
    pickup_every_steps: int = 400

    # --- Training only ---
    max_episode_steps: int = 5000

    # --- Level switches ---
    @property
    def shooting_enabled(self) -> bool:
        return self.level >= 2

    @property
    def ammo_limited(self) -> bool:
        return self.level >= 3

    @property
    def hunters_enabled(self) -> bool:
        return self.level >= 4

    @property
    def shields_enabled(self) -> bool:
        return self.level >= 4
    
    

@dataclass
class RewardConfig:
    """The reward weights. Kept apart from the game rules so we can tune them freely."""

    survive: float = 0.01              # every step the game is still running
    drifter_killed: float = 1.0
    hunter_killed: float = 2.0
    life_lost: float = -5.0
    shield_collected: float = 0.5