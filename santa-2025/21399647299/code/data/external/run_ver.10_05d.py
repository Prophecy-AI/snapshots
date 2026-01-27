import pygame
import pymunk
import pymunk.pygame_util
import random

# --- Constants ---
WIDTH, HEIGHT = 800, 600
FPS = 60

# --- Visualization Constants ---
PIXELS_PER_UNIT = 100.0
GRID_MINOR = 0.2
GRID_MAJOR = 1.0

# Colors
COLOR_BG = (245, 245, 245)
COLOR_GRID_MINOR = (220, 220, 220)
COLOR_GRID_MAJOR = (150, 150, 150)
COLOR_AXIS = (140, 0, 210)
COLOR_UNIFIED_TREE = (144, 238, 144)
COLOR_TREE_OUTLINE = (0, 0, 0)
COLOR_WALL = (50, 50, 50)

# Dimensions
WALL_THICKNESS = 60
WALL_LENGTH = 1000
MAX_OPEN_DIST = 350

# Physics Config
PRESS_FORCE = 800000
RETRACT_FORCE = 40000
CENTERING_SPEED = 120
HOLD_DURATION = 1.0

# Agitation Config
VIBRATION_STRENGTH = 200.0
ROTATION_JITTER = 0.5

# [NEW] Cooling Thresholds (Distance between Left/Right walls)
# Width > 550px: 100% Vibration
# Width 550px -> 320px: Linear drop to 0%
# Width < 320px: 0% Vibration (Settling)
WIDTH_HOT_START = 550.0
WIDTH_COLD_END = 320.0

# Collision Categories
CAT_WALL = 0b01
CAT_TREE = 0b10


class HydraulicPressSim:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Hydraulic Press - Distance Based Cooling")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 12)
        self.font_ui = pygame.font.SysFont("Arial", 24)

        self.space = pymunk.Space()
        self.space.gravity = (0, 0)
        self.space.damping = 0.5
        self.space.iterations = 80

        self.walls = []
        self.objects = []

        self.state = "SQUEEZE"
        self.state_timer = 0
        self.current_temp = 0.0  # For UI display

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
            start_dist = 300
            start_pos = (center[0] + dx * start_dist, center[1] + dy * start_dist)
            limit_dist_px = MAX_OPEN_DIST
            extended_limit = (center[0] + dx * limit_dist_px * 1.5, center[1] + dy * limit_dist_px * 1.5)

            mass = 500
            moment = float('inf')
            body = pymunk.Body(mass, moment)
            body.position = start_pos

            if dx == 0:
                size = (WALL_LENGTH, WALL_THICKNESS)
            else:
                size = (WALL_THICKNESS, WALL_LENGTH)

            shape = pymunk.Poly.create_box(body, size)
            shape.elasticity = 0.0
            shape.friction = 0.5
            shape.filter = pymunk.ShapeFilter(categories=CAT_WALL, mask=CAT_TREE)

            joint = pymunk.GrooveJoint(static_center, body, center, extended_limit, (0, 0))
            joint.error_bias = 0.001

            self.space.add(body, shape, joint)
            self.walls.append(body)

    def spawn_trees(self):
        for i in range(12):
            self.create_complex_tree_at_random_pos()

    def create_complex_tree_at_random_pos(self):
        s = PIXELS_PER_UNIT

        trunk_w = 0.15 * s
        trunk_h = 0.2 * s
        base_w = 0.7 * s
        mid_w = 0.4 * s
        top_w = 0.25 * s

        tip_y = 0.8 * s
        tier_1_y = 0.5 * s
        tier_2_y = 0.25 * s
        base_y = 0.0 * s
        trunk_bottom_y = -trunk_h

        mass = 1
        moment = pymunk.moment_for_box(mass, (base_w, tip_y - trunk_bottom_y))
        body = pymunk.Body(mass, moment)

        spawn_spread = 150
        body.position = (WIDTH / 2 + random.randint(-spawn_spread, spawn_spread),
                         HEIGHT / 2 + random.randint(-spawn_spread, spawn_spread))
        body.angle = random.random() * 6.28

        self.space.add(body)
        self.objects.append(body)

        verts_top = [(0, tip_y), (top_w / 2, tier_1_y), (-top_w / 2, tier_1_y)]
        verts_mid = [(top_w / 4, tier_1_y), (mid_w / 2, tier_2_y), (-mid_w / 2, tier_2_y), (-top_w / 4, tier_1_y)]
        verts_bot = [(mid_w / 4, tier_2_y), (base_w / 2, base_y), (-base_w / 2, base_y), (-mid_w / 4, tier_2_y)]
        verts_trunk = [(trunk_w / 2, base_y), (trunk_w / 2, trunk_bottom_y), (-trunk_w / 2, trunk_bottom_y),
                       (-trunk_w / 2, base_y)]

        def add_poly(vertices):
            shape = pymunk.Poly(body, vertices)
            shape.elasticity = 0.0
            shape.friction = 0.3
            shape.filter = pymunk.ShapeFilter(categories=CAT_TREE, mask=CAT_WALL | CAT_TREE)
            self.space.add(shape)

        add_poly(verts_trunk)
        add_poly(verts_bot)
        add_poly(verts_mid)
        add_poly(verts_top)

        self.perimeter_local = [
            (-trunk_w / 2, trunk_bottom_y), (-trunk_w / 2, base_y), (-base_w / 2, base_y),
            (-mid_w / 4, tier_2_y), (-mid_w / 2, tier_2_y), (-top_w / 4, tier_1_y),
            (-top_w / 2, tier_1_y), (0, tip_y), (top_w / 2, tier_1_y),
            (top_w / 4, tier_1_y), (mid_w / 2, tier_2_y), (mid_w / 4, tier_2_y),
            (base_w / 2, base_y), (trunk_w / 2, base_y), (trunk_w / 2, trunk_bottom_y),
        ]
        body.custom_perimeter = self.perimeter_local

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

    def apply_brownian_motion(self, scale=1.0):
        """
        scale: 0.0 to 1.0 (Temperature factor)
        """
        if scale <= 0.01: return

        for body in self.objects:
            # Scale the impulse and torque by the temperature
            impulse = pymunk.Vec2d(random.uniform(-1, 1), random.uniform(-1, 1)) * VIBRATION_STRENGTH * scale
            body.apply_impulse_at_local_point(impulse)

            torque = random.uniform(-ROTATION_JITTER, ROTATION_JITTER) * 10000 * scale
            body.torque = torque

    def logic_cycle(self):
        wall_T, wall_B = self.walls[0], self.walls[1]
        wall_L, wall_R = self.walls[2], self.walls[3]

        # Calculate current squeeze width
        width_curr = wall_R.position.x - wall_L.position.x
        center = pymunk.Vec2d(WIDTH / 2, HEIGHT / 2)

        if self.state == "SQUEEZE":
            # 1. Apply Press Force
            total_velocity = 0
            for wall in self.walls:
                direction = (center - wall.position).normalized()
                wall.apply_force_at_local_point(direction * PRESS_FORCE)
                total_velocity += wall.velocity.length

            # 2. Calculate "Temperature" based on Distance
            # If width > 550, Temp = 1.0
            # If width < 320, Temp = 0.0
            # Linear in between
            if width_curr > WIDTH_HOT_START:
                self.current_temp = 1.0
            elif width_curr < WIDTH_COLD_END:
                self.current_temp = 0.0
            else:
                # Normalization formula
                range_span = WIDTH_HOT_START - WIDTH_COLD_END
                self.current_temp = (width_curr - WIDTH_COLD_END) / range_span

            # 3. Apply Agitation based on Temperature
            self.apply_brownian_motion(scale=self.current_temp)

            # 4. Stop Condition
            # Velocity must be low AND Temperature must be low (Settled)
            # This prevents premature stopping when walls are still far out but vibrating hard
            if total_velocity < 2.0 or self.current_temp < 0.005:
                self.state_timer += 1
                if self.state_timer > 30:  # Confirm settlement (0.5s)

                    # Record
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
                self.state = "HOLD"
                self.state_timer = 0

        elif self.state == "HOLD":
            for w in self.walls: w.velocity = (0, 0)
            self.state_timer += 1
            if self.state_timer > FPS * HOLD_DURATION:
                self.state = "EXPAND"

        elif self.state == "EXPAND":

            self.apply_brownian_motion(scale=1)
            for wall in self.walls:
                dist = (wall.position - center).length
                if dist < 220:
                    direction = (wall.position - center).normalized()
                    wall.apply_force_at_local_point(direction * RETRACT_FORCE)
                else:
                    self.state = "SHUFFLE"

        elif self.state == "SHUFFLE":
            for obj in self.objects:
                obj.angular_velocity += random.uniform(-1, 1)
            self.state = "SQUEEZE"

    def draw_grid(self):
        self.screen.fill(COLOR_BG)
        cx, cy = WIDTH // 2, HEIGHT // 2
        minor_px = PIXELS_PER_UNIT * GRID_MINOR

        num_lines_x = int(cx / minor_px) + 1
        num_lines_y = int(cy / minor_px) + 1

        for i in range(-num_lines_x, num_lines_x + 1):
            px = cx + i * minor_px
            is_major = (i % 5 == 0)
            is_axis = (i == 0)
            color = COLOR_AXIS if is_axis else (COLOR_GRID_MAJOR if is_major else COLOR_GRID_MINOR)
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
            color = COLOR_AXIS if is_axis else (COLOR_GRID_MAJOR if is_major else COLOR_GRID_MINOR)
            width = 2 if (is_major or is_axis) else 1
            pygame.draw.line(self.screen, color, (0, py), (WIDTH, py), width)
            if is_major and not is_axis:
                val = -i * GRID_MINOR
                label = self.font.render(f"{val:.1f}", True, (80, 80, 80))
                self.screen.blit(label, (cx + 4, py - 12))

    def draw_shapes_custom(self):
        for body in self.objects:
            for shape in body.shapes:
                world_points = [body.local_to_world(v) for v in shape.get_vertices()]
                pts = [(int(p.x), int(p.y)) for p in world_points]
                pygame.draw.polygon(self.screen, COLOR_UNIFIED_TREE, pts)

        for body in self.objects:
            if hasattr(body, "custom_perimeter"):
                world_perimeter = [body.local_to_world(v) for v in body.custom_perimeter]
                pts = [(int(p.x), int(p.y)) for p in world_perimeter]
                pygame.draw.lines(self.screen, COLOR_TREE_OUTLINE, True, pts, 2)

        for body in self.walls:
            for shape in body.shapes:
                world_points = [body.local_to_world(v) for v in shape.get_vertices()]
                pts = [(int(p.x), int(p.y)) for p in world_points]
                pygame.draw.polygon(self.screen, COLOR_WALL, pts)

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

        # Display Squeeze status + Temperature
        status_text = f"State: {self.state}"
        if self.state == "SQUEEZE":
            temp_pct = int(self.current_temp * 100)
            status_text += f" (Temp: {temp_pct}%)"

        status = self.font_ui.render(status_text, True, (0, 0, 0))
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
            self.draw_shapes_custom()
            self.draw_info()

            pygame.display.flip()
            self.clock.tick(FPS)


if __name__ == "__main__":
    HydraulicPressSim().run()