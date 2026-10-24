import random

import pygame

from .game import DRIFTER, HUNTER

# Colors (R, G, B)
BACKGROUND = (10, 12, 30)
SHIP_COLOR = (80, 200, 255)
DRIFTER_COLOR = (230, 70, 70)
HUNTER_COLOR = (255, 160, 40)
TEXT_COLOR = (235, 235, 235)
HITBOX_COLOR = (80, 255, 120)
BULLET_COLOR = (255, 230, 90)
SHIELD_COLOR = (90, 255, 200)
HUNTER_HURT_COLOR = (255, 215, 140)


class Renderer:
    """Draws the game on a window. It only READS the game, it never changes it."""

    def __init__(self, cfg, fps=60):
        pygame.init()
        self.cfg = cfg
        self.fps = fps                      # 0 = no speed limit
        self.show_hitboxes = False

        size = cfg.arena_size
        self.screen = pygame.display.set_mode((size, size))
        pygame.display.set_caption("Space Survival Shooter")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 22)
        self.big_font = pygame.font.SysFont("consolas", 54)

        # Decorative stars. Own random generator: it must NOT touch the game's randomness.
        star_rng = random.Random(0)
        self.stars = [
            (star_rng.randint(0, size), star_rng.randint(0, size), star_rng.randint(60, 160))
            for _ in range(70)
        ]

    # ------------------------------------------------------------------
    def draw(self, game):
        """Draw one frame. Things drawn later appear on top of earlier ones."""
        screen = self.screen
        screen.fill(BACKGROUND)

        for x, y, brightness in self.stars:
            pygame.draw.circle(screen, (brightness,) * 3, (x, y), 1)

        for enemy in game.enemies:
            if enemy.kind == DRIFTER:
                color = DRIFTER_COLOR
            elif enemy.hp >= self.cfg.hunter_hp:
                color = HUNTER_COLOR
            else:
                color = HUNTER_HURT_COLOR        # a wounded Hunter looks paler
                
            pos = (int(enemy.x), int(enemy.y))
            pygame.draw.circle(screen, color, pos, int(self.cfg.enemy_radius))
            if self.show_hitboxes:
                pygame.draw.circle(screen, HITBOX_COLOR, pos, int(self.cfg.enemy_radius), 1)
        for bullet in game.bullets:
            pos = (int(bullet.x), int(bullet.y))
            pygame.draw.circle(screen, BULLET_COLOR, pos, int(self.cfg.bullet_radius))
            if self.show_hitboxes:
                pygame.draw.circle(screen, HITBOX_COLOR, pos, int(self.cfg.bullet_radius), 1)
        
        for pickup in game.pickups:
            pos = (int(pickup.x), int(pickup.y))
            pygame.draw.circle(screen, SHIELD_COLOR, pos, int(self.cfg.pickup_radius), 3)
            pygame.draw.circle(screen, SHIELD_COLOR, pos, 3)
            if self.show_hitboxes:
                pygame.draw.circle(screen, HITBOX_COLOR, pos, int(self.cfg.pickup_radius), 1)
        self._draw_ship(game)
        
        self._draw_hud(game)

        if game.game_over:
            self._draw_game_over()

        pygame.display.flip()               # show the finished frame
        self.clock.tick(self.fps)           # wait so the game runs at `fps`
        if game.shield:
            center = (int(game.ship_x), int(game.ship_y))
            pygame.draw.circle(self.screen, SHIELD_COLOR, center, int(self.cfg.ship_radius) + 8, 2)
    # ------------------------------------------------------------------
    def _draw_ship(self, game):
        # Blink while the ship is protected: hidden every other 4 steps
        if game.invulnerable > 0 and (game.invulnerable // 4) % 2 == 0:
            return

        x, y, r = game.ship_x, game.ship_y, self.cfg.ship_radius
        triangle = [(x, y - r), (x - r, y + r), (x + r, y + r)]   # points up
        pygame.draw.polygon(self.screen, SHIP_COLOR, triangle)

        if self.show_hitboxes:
            pygame.draw.circle(self.screen, HITBOX_COLOR, (int(x), int(y)), int(r), 1)

    def _draw_hud(self, game):
        kills = game.stats["drifters_killed"] + game.stats["hunters_killed"]
        lives = self.font.render(f"Lives: {game.lives}", True, TEXT_COLOR)
        kills_text = self.font.render(f"Kills: {kills}", True, TEXT_COLOR)
        steps = self.font.render(f"Steps: {game.steps}", True, TEXT_COLOR)
        self.screen.blit(lives, (10, 8))
        self.screen.blit(kills_text, (10, 34))
        self.screen.blit(steps, (self.cfg.arena_size - steps.get_width() - 10, 8))

        if self.cfg.ammo_limited:
            label = f"Ammo: {game.ammo}/{self.cfg.max_ammo}"
        else:
            label = "Ammo: unlimited"
        self.screen.blit(self.font.render(label, True, TEXT_COLOR), (10, 60))

        if self.cfg.ammo_limited:
            bar_width = 120
            fraction = game.ammo / self.cfg.max_ammo
            color = BULLET_COLOR if game.ammo > 0 else DRIFTER_COLOR
            pygame.draw.rect(self.screen, TEXT_COLOR, (10, 88, bar_width, 10), 1)
            pygame.draw.rect(self.screen, color, (10, 88, int(bar_width * fraction), 10))
        if self.cfg.shields_enabled:
            text = "Shield: ON" if game.shield else "Shield: off"
            color = SHIELD_COLOR if game.shield else TEXT_COLOR
            self.screen.blit(self.font.render(text, True, color), (10, 106))
    def _draw_game_over(self):
        size = self.cfg.arena_size
        title = self.big_font.render("GAME OVER", True, DRIFTER_COLOR)
        hint = self.font.render("R = restart   ESC = quit", True, TEXT_COLOR)
        self.screen.blit(title, (size // 2 - title.get_width() // 2, size // 2 - 50))
        self.screen.blit(hint, (size // 2 - hint.get_width() // 2, size // 2 + 20))

    def close(self):
        pygame.quit()