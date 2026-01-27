import pygame
import pymunk
import pymunk.pygame_util
import random

# --- Constants ---
WIDTH, HEIGHT = 800, 600
FPS = 60

# --- Visualization Constants ---
PIXELS_PER_UNIT = 40.0
GRID_MINOR = 0.2
GRID_MAJOR = 1.0

# Dimensions
WALL_THICKNESS = 40
WALL_LENGTH = 800
MAX_OPEN_DIST = 280

# Physics Config
PRESS_FORCE = 60000
RETRACT_FORCE = 30000
CENTERING_SPEED = 100
HOLD_DURATION = 2.0  # Seconds to pause and admire the result

# Collision Categories
CAT_WALL = 0b01
CAT_TREE = 0b10


class HydraulicPressSim:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Hydraulic Press - Pause & Admire")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 12)
        self.font_ui = pygame.font.SysFont("Arial", 24)

        self.space = pymunk.Space()
        self.space.gravity = (0, 0)
        self.space.damping = 0.8
        self.space.iterations = 60

        self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)
        self.draw_options.flags = pymunk.pygame_util.DrawOptions.DRAW_SHAPES

        self.walls = []
        self.objects = []

        # State Machine
        self.state = "SQUEEZE"
        self.state_timer = 0

        # Data Recording
        self.last_recorded_area = 0.0
        self.min_tree_area = float('inf')
        self.best_dims_px = None

        self.build_walls()
        self.spawn_trees()

    def build_walls(self):
        static_center = self.space.static_body
        center = (WIDTH / 2, HEIGHT / 2)

        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]

        for dx, dy in directions:
            start_dist = 200
            start_pos = (center[0] + dx * start_dist, center[1] + dy * start_dist)
            limit_dist_px = MAX_OPEN_DIST
            extended_limit = (center[0] + dx * limit_dist_px * 1.5, center[1] + dy * limit_dist_px * 1.5)

            mass = 200
            moment = float('inf')
            body = pymunk.Body(mass, moment)
            body.position = start_pos

            if dx == 0:
                size = (WALL_LENGTH, WALL_THICKNESS)
            else:
                size = (WALL_THICKNESS, WALL_LENGTH)

            shape = pymunk.Poly.create_box(body, size)
            shape.elasticity = 0.0
            shape.friction = 0.8
            shape.color = (50, 50, 50, 255)
            shape.filter = pymunk.ShapeFilter(categories=CAT_WALL, mask=CAT_TREE)

            joint = pymunk.GrooveJoint(static_center, body, center, extended_limit, (0, 0))
            joint.error_bias = 0.001

            self.space.add(body, shape, joint)
            self.walls.append(body)

    def spawn_trees(self):
        for i in range(15):
            mass = 1
            size = 35
            points = [(-size // 2, -size // 2), (size // 2, -size // 2), (0, size // 2)]
            moment = pymunk.moment_for_poly(mass, points)
            body = pymunk.Body(mass, moment)

            body.position = (WIDTH / 2 + random.randint(-40, 40), HEIGHT / 2 + random.randint(-40, 40))
            body.angle = random.random() * 6.28

            shape = pymunk.Poly(body, points)
            shape.elasticity = 0.1
            shape.friction = 0.6
            shape.color = (34, 139, 34, 255)
            shape.filter = pymunk.ShapeFilter(categories=CAT_TREE, mask=CAT_WALL | CAT_TREE)

            self.space.add(body, shape)
            self.objects.append(body)

    def get_tree_bounds(self):
        min_x, min_y = float('inf'), float('inf')
        max_x, max_y = float('-inf'), float('-inf')

        for body in self.objects:
            for shape in body.shapes:
                bb = shape.bb
                if bb.left < min_x: min_x = bb.left
                if bb.bottom < min_y: min_y = bb.bottom
                if bb.right > max_x: max_x = bb.right
                if bb.top > max_y: max_y = bb.top

        width = max_x - min_x
        height = max_y - min_y
        return width, height

    def logic_cycle(self):
        wall_T, wall_B = self.walls[0], self.walls[1]
        wall_L, wall_R = self.walls[2], self.walls[3]

        width_curr = wall_R.position.x - wall_L.position.x
        center = pymunk.Vec2d(WIDTH / 2, HEIGHT / 2)

        # --- STATE MACHINE ---

        if self.state == "SQUEEZE":
            total_velocity = 0
            for wall in self.walls:
                direction = (center - wall.position).normalized()
                wall.apply_force_at_local_point(direction * PRESS_FORCE)
                total_velocity += wall.velocity.length

            if total_velocity < 2.0 and width_curr < 350:
                self.state_timer += 1
                if self.state_timer > 60:
                    # Record Results
                    tree_w, tree_h = self.get_tree_bounds()
                    area_units = (tree_w * tree_h) / (PIXELS_PER_UNIT * PIXELS_PER_UNIT)
                    self.last_recorded_area = area_units

                    if area_units < self.min_tree_area:
                        self.min_tree_area = area_units
                        self.best_dims_px = (tree_w, tree_h)

                    self.state = "CENTER"
                    self.state_timer = 0
            else:
                self.state_timer = 0

        elif self.state == "CENTER":
            mid_x = (wall_L.position.x + wall_R.position.x) / 2
            drift_x = mid_x - WIDTH / 2
            mid_y = (wall_T.position.y + wall_B.position.y) / 2
            drift_y = mid_y - HEIGHT / 2

            vx, vy = 0, 0
            if abs(drift_x) > 2.0: vx = -CENTERING_SPEED if drift_x > 0 else CENTERING_SPEED
            if abs(drift_y) > 2.0: vy = -CENTERING_SPEED if drift_y > 0 else CENTERING_SPEED

            wall_L.velocity = (vx, 0);
            wall_R.velocity = (vx, 0)
            wall_T.velocity = (0, vy);
            wall_B.velocity = (0, vy)

            if vx == 0 and vy == 0:
                # DONE CENTERING -> GO TO HOLD
                self.state = "HOLD"
                self.state_timer = 0

        # --- NEW STATE: HOLD ---
        elif self.state == "HOLD":
            # Force walls to stay still (Lock them in place)
            # This prevents them from drifting while we admire the result
            for w in self.walls:
                w.velocity = (0, 0)

            self.state_timer += 1
            # Wait for 2 seconds (60 frames * 2)
            if self.state_timer > FPS * HOLD_DURATION:
                self.state = "EXPAND"
        # -----------------------

        elif self.state == "EXPAND":
            for wall in self.walls:
                dist = (wall.position - center).length
                if dist < MAX_OPEN_DIST - 10:
                    direction = (wall.position - center).normalized()
                    wall.apply_force_at_local_point(direction * RETRACT_FORCE)

            if width_curr > (MAX_OPEN_DIST * 1.8):
                self.state = "SHUFFLE"

        elif self.state == "SHUFFLE":
            for obj in self.objects:
                impulse = pymunk.Vec2d(random.uniform(-40, 40), random.uniform(-40, 40))
                obj.apply_impulse_at_local_point(impulse)
                obj.angular_velocity += random.uniform(-2, 2)
            self.state = "SQUEEZE"

    def draw_grid(self):
        BG_COLOR = (245, 245, 245)
        MINOR_COLOR = (220, 220, 220)
        MAJOR_COLOR = (150, 150, 150)
        AXIS_COLOR = (140, 0, 210)

        self.screen.fill(BG_COLOR)
        cx, cy = WIDTH // 2, HEIGHT // 2
        minor_px = PIXELS_PER_UNIT * GRID_MINOR

        num_lines_x = int(cx / minor_px) + 1
        num_lines_y = int(cy / minor_px) + 1

        for i in range(-num_lines_x, num_lines_x + 1):
            px = cx + i * minor_px
            is_major = (i % 5 == 0)
            is_axis = (i == 0)
            color = AXIS_COLOR if is_axis else (MAJOR_COLOR if is_major else MINOR_COLOR)
            width = 2 if (is_major or is_axis) else 1
            pygame.draw.line(self.screen, color, (px, 0), (px, HEIGHT), width)
            if is_major and not is_axis:
                val = i * GRID_MINOR
                label = self.font.render(f"{val:.1f}", True, (80, 80, 80))
                self.screen.blit(label, (px + 2, cy + 2))

        for i in range(-num_lines_y, num_lines_y + 1):
            py = cy + i * minor_px
            is_major = (i % 5 == 0)
            is_axis = (i == 0)
            color = AXIS_COLOR if is_axis else (MAJOR_COLOR if is_major else MINOR_COLOR)
            width = 2 if (is_major or is_axis) else 1
            pygame.draw.line(self.screen, color, (0, py), (WIDTH, py), width)
            if is_major and not is_axis:
                val = -i * GRID_MINOR
                label = self.font.render(f"{val:.1f}", True, (80, 80, 80))
                self.screen.blit(label, (cx + 4, py - 12))

    def draw_best_area_rect(self):
        if self.best_dims_px is not None:
            w, h = self.best_dims_px
            s = pygame.Surface((int(w), int(h)), pygame.SRCALPHA)
            s.fill((255, 0, 0, 100))
            pygame.draw.rect(s, (200, 0, 0, 255), s.get_rect(), 3)
            rect = s.get_rect(center=(WIDTH // 2, HEIGHT // 2))
            self.screen.blit(s, rect)

    def draw_info(self):
        pygame.draw.rect(self.screen, (255, 255, 255), (15, 15, 300, 90))
        pygame.draw.rect(self.screen, (0, 0, 0), (15, 15, 300, 90), 2)

        status = self.font_ui.render(f"State: {self.state}", True, (0, 0, 0))
        best_val = self.min_tree_area if self.min_tree_area != float('inf') else 0.0
        best_txt = self.font_ui.render(f"Best Tree Area: {best_val:.4f}", True, (200, 0, 0))
        curr_txt = self.font_ui.render(f"Last Recorded: {self.last_recorded_area:.4f}", True, (0, 0, 0))

        self.screen.blit(status, (25, 20))
        self.screen.blit(best_txt, (25, 50))
        self.screen.blit(curr_txt, (25, 80))

    def run(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return

            self.logic_cycle()

            steps = 10
            for _ in range(steps):
                self.space.step(1.0 / FPS / steps)

            self.draw_grid()
            self.draw_best_area_rect()
            self.space.debug_draw(self.draw_options)
            self.draw_info()

            pygame.display.flip()
            self.clock.tick(FPS)


if __name__ == "__main__":
    HydraulicPressSim().run()