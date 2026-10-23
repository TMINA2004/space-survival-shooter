import time

from space_shooter.game import Game, STAY, UP, DOWN, LEFT, RIGHT

game = Game()
print("Start position:", game.ship_x, game.ship_y)

# Test 1: moving UP 10 times decreases y by 60
for _ in range(10):
    game.step(UP)

assert game.ship_y == 540 - 60, "UP is wrong"
print("Test 1 OK: UP works")

# Test 2: the ship stops at the left wall
for _ in range(200):
    game.step(LEFT)

assert game.ship_x == game.cfg.ship_radius, "left wall is wrong"
print("Test 2 OK: left wall stops the ship at x =", game.ship_x)

# Test 3: the ship stops at the bottom-right corner
for _ in range(300):
    game.step(RIGHT)

for _ in range(300):
    game.step(DOWN)

corner = game.cfg.arena_size - game.cfg.ship_radius

assert game.ship_x == corner and game.ship_y == corner, "corner is wrong"
print("Test 3 OK: bottom-right corner =", game.ship_x, game.ship_y)

# Test 4: reset brings the ship back
game.reset()

assert (game.ship_x, game.ship_y) == (300, 540), "reset is wrong"
print("Test 4 OK: reset works")

# Speed test: how many steps per second without graphics?
start = time.time()

for _ in range(200_000):
    game.step(STAY)

seconds = time.time() - start

print(f"Speed: {200_000 / seconds:,.0f} steps per second")