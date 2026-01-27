import pygame
import pymunk
import pymunk.pygame_util
import random

# --- Constants ---
WIDTH, HEIGHT = 800, 600
FPS = 60

# Dimensions
WALL_THICKNESS = 40
WALL_LENGTH = 800  # Long enough to always overlap corners
MAX_OPEN_DIST = 280  # How far from center walls can go (The Hard Limit)

# Physics
PRESS_FORCE = 50000
RETRACT_FORCE = 30000

# Collision Categories
CAT_WALL = 0b01
CAT_TREE = 0b10


class HydraulicPressSim:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Hydraulic Press - Fixed Limits")
        self.clock = pygame.time.Clock()

        self.space = pymunk.Space()
        self.space.gravity = (0, 0)
        self.space.damping = 0.8
        self.space.iterations = 60

        self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)

        self.walls = []
        self.objects = []

        self.state = "SQUEEZE"
        self.state_timer = 0
        self.min_area_found = float('inf')

        self.build_walls()
        self.spawn_trees()

    def build_walls(self):
        static_center = self.space.static_body
        center = (WIDTH / 2, HEIGHT / 2)

        # We define directions for the 4 walls
        # (Start Offset Multiplier X, Start Offset Multiplier Y)
        directions = [
            (0, -1),  # Top
            (0, 1),  # Bottom
            (-1, 0),  # Left
            (1, 0)  # Right
        ]

        for dx, dy in directions:
            # 1. Calculate positions
            # Start slightly inward so we don't start against the hard stop
            start_dist = 200

            # Position = Center + (Direction * Distance)
            start_pos = (center[0] + dx * start_dist, center[1] + dy * start_dist)

            # Limit Position (The end of the rail)
            limit_pos = (center[0] + dx * MAX_OPEN_DIST, center[1] + dy * MAX_OPEN_DIST)

            # 2. Create Body
            mass = 100
            moment = float('inf')  # Infinite moment = No Rotation
            body = pymunk.Body(mass, moment)
            body.position = start_pos

            # 3. Create Shape
            if dx == 0:  # Vertical movement (Top/Bottom walls)
                size = (WALL_LENGTH, WALL_THICKNESS)
            else:  # Horizontal movement (Left/Right walls)
                size = (WALL_THICKNESS, WALL_LENGTH)

            shape = pymunk.Poly.create_box(body, size)
            shape.elasticity = 0.0
            shape.friction = 0.5
            shape.color = (50, 50, 50, 255)
            # Filter: Walls ignore other Walls (CAT_WALL), but hit Trees (CAT_TREE)
            shape.filter = pymunk.ShapeFilter(categories=CAT_WALL, mask=CAT_TREE)

            # 4. Create Rail (GrooveJoint)
            # The Groove goes from Center -> Limit Position
            # The Body is constrained to this line. It literally cannot leave it.
            joint = pymunk.GrooveJoint(static_center, body, center, limit_pos, (0, 0))
            joint.error_bias = 0.001

            self.space.add(body, shape, joint)
            self.walls.append(body)

    def spawn_trees(self):
        # Spawn some triangles
        for i in range(15):
            mass = 1
            size = 35
            points = [(-size // 2, -size // 2), (size // 2, -size // 2), (0, size // 2)]
            moment = pymunk.moment_for_poly(mass, points)
            body = pymunk.Body(mass, moment)

            # Random spawn near center
            body.position = (WIDTH / 2 + random.randint(-40, 40), HEIGHT / 2 + random.randint(-40, 40))
            body.angle = random.random() * 6.28

            shape = pymunk.Poly(body, points)
            shape.elasticity = 0.1
            shape.friction = 0.5
            shape.color = (34, 139, 34, 255)
            # Trees hit Everything (Walls | Trees)
            shape.filter = pymunk.ShapeFilter(categories=CAT_TREE, mask=CAT_WALL | CAT_TREE)

            self.space.add(body, shape)
            self.objects.append(body)

    def logic_cycle(self):
        # Calculate Current Gap Size
        w_curr = self.walls[3].position.x - self.walls[2].position.x
        current_area = w_curr * w_curr  # Approximate area

        center = pymunk.Vec2d(WIDTH / 2, HEIGHT / 2)

        if self.state == "SQUEEZE":
            total_velocity = 0
            for wall in self.walls:
                # Push towards center
                direction = (center - wall.position).normalized()
                wall.apply_force_at_local_point(direction * PRESS_FORCE)
                total_velocity += wall.velocity.length

            # Stop Condition: Slow movement AND tight squeeze
            if total_velocity < 2.0 and w_curr < 350:
                self.state_timer += 1
                if self.state_timer > 60:
                    if current_area < self.min_area_found:
                        self.min_area_found = current_area
                    self.state = "EXPAND"
                    self.state_timer = 0
            else:
                self.state_timer = 0

        elif self.state == "EXPAND":
            for wall in self.walls:
                # Check distance from center
                dist = (wall.position - center).length

                # Only pull back if we haven't hit the limit yet
                # This prevents slamming into the end of the rail
                if dist < MAX_OPEN_DIST - 10:
                    direction = (wall.position - center).normalized()
                    wall.apply_force_at_local_point(direction * RETRACT_FORCE)

            # If we are mostly open, switch to shuffle
            # MAX_OPEN_DIST * 2 is roughly the full width, minus some buffer
            if w_curr > (MAX_OPEN_DIST * 1.8):
                self.state = "SHUFFLE"

        elif self.state == "SHUFFLE":
            for obj in self.objects:
                impulse = pymunk.Vec2d(random.uniform(-40, 40), random.uniform(-40, 40))
                obj.apply_impulse_at_local_point(impulse)
                obj.angular_velocity += random.uniform(-2, 2)

            self.state = "SQUEEZE"

    def draw_rails(self):
        # Helper to visualize where the walls are allowed to go
        center = (WIDTH // 2, HEIGHT // 2)
        # Draw a cross showing the limits
        pygame.draw.line(self.screen, (200, 200, 200), (center[0], center[1] - MAX_OPEN_DIST),
                         (center[0], center[1] + MAX_OPEN_DIST), 2)
        pygame.draw.line(self.screen, (200, 200, 200), (center[0] - MAX_OPEN_DIST, center[1]),
                         (center[0] + MAX_OPEN_DIST, center[1]), 2)

    def draw_info(self):
        font = pygame.font.SysFont("Arial", 24)
        status = font.render(f"State: {self.state}", True, (0, 0, 0))
        best = font.render(f"Best Area: {int(self.min_area_found) if self.min_area_found != float('inf') else 0}", True,
                           (0, 0, 0))
        self.screen.blit(status, (20, 20))
        self.screen.blit(best, (20, 50))

    def run(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return

            self.logic_cycle()

            steps = 10
            for _ in range(steps):
                self.space.step(1.0 / FPS / steps)

            self.screen.fill((240, 240, 240))

            # Draw the guide rails first
            self.draw_rails()

            self.space.debug_draw(self.draw_options)
            self.draw_info()

            pygame.display.flip()
            self.clock.tick(FPS)


if __name__ == "__main__":
    HydraulicPressSim().run()