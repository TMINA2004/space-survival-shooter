import pygame

from space_shooter.config import Config
from space_shooter.game import Game, STAY, UP, DOWN, LEFT, RIGHT, make_action
from space_shooter.renderer import Renderer
import sys


def read_action():
    """Turn the keyboard into ONE action (0-9): a move plus an optional shot."""
    keys = pygame.key.get_pressed()

    move = STAY
    if keys[pygame.K_UP] or keys[pygame.K_w] or keys[pygame.K_z]:
        move = UP
    elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
        move = DOWN
    elif keys[pygame.K_LEFT] or keys[pygame.K_a] or keys[pygame.K_q]:
        move = LEFT
    elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
        move = RIGHT

    shoot = keys[pygame.K_SPACE]
    return make_action(move, shoot)


def main():
    level = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    cfg = Config(level=level)
    game = Game(cfg)
    renderer = Renderer(cfg, fps=60)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    game.reset()
                elif event.key == pygame.K_h:
                    renderer.show_hitboxes = not renderer.show_hitboxes

        game.step(read_action())
        renderer.draw(game)

    renderer.close()


if __name__ == "__main__":
    main()