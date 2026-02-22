# env import
import gymnasium as gym
from pymunk.space_debug_draw_options import SpaceDebugColor
from pymunk.vec2d import Vec2d
from typing import Tuple, Sequence
import pygame
import pymunk
import numpy as np
import shapely.geometry as sg
import cv2
import pymunk.pygame_util
import collections

from scipy.spatial.distance import cdist
import torch


_POPCOUNT_TABLE8 = [bin(i).count('1') for i in range(256)]

positive_y_is_up: bool = False
"""Make increasing values of y point upwards.

When True::

    y
    ^
    |      . (3, 3)
    |
    |   . (2, 2)
    |
    +------ > x

When False::

    +------ > x
    |
    |   . (2, 2)
    |
    |      . (3, 3)
    v
    y

"""

class DrawOptions(pymunk.SpaceDebugDrawOptions):
    def __init__(self, surface: pygame.Surface) -> None:
        """Draw a pymunk.Space on a pygame.Surface object.

        Typical usage::

        >>> import pymunk
        >>> surface = pygame.Surface((10,10))
        >>> space = pymunk.Space()
        >>> options = pymunk.pygame_util.DrawOptions(surface)
        >>> space.debug_draw(options)

        You can control the color of a shape by setting shape.color to the color
        you want it drawn in::

        >>> c = pymunk.Circle(None, 10)
        >>> c.color = pygame.Color("pink")

        See pygame_util.demo.py for a full example

        Since pygame uses a coordinate system where y points down (in contrast
        to many other cases), you either have to make the physics simulation
        with Pymunk also behave in that way, or flip everything when you draw.

        The easiest is probably to just make the simulation behave the same
        way as Pygame does. In that way all coordinates used are in the same
        orientation and easy to reason about::

        >>> space = pymunk.Space()
        >>> space.gravity = (0, -1000)
        >>> body = pymunk.Body()
        >>> body.position = (0, 0) # will be positioned in the top left corner
        >>> space.debug_draw(options)

        To flip the drawing its possible to set the module property
        :py:data:`positive_y_is_up` to True. Then the pygame drawing will flip
        the simulation upside down before drawing::

        >>> positive_y_is_up = True
        >>> body = pymunk.Body()
        >>> body.position = (0, 0)
        >>> # Body will be position in bottom left corner

        :Parameters:
                surface : pygame.Surface
                    Surface that the objects will be drawn on
        """
        self.surface = surface
        super(DrawOptions, self).__init__()

    def draw_circle(
        self,
        pos: Vec2d,
        angle: float,
        radius: float,
        outline_color: SpaceDebugColor,
        fill_color: SpaceDebugColor,
    ) -> None:
        p = to_pygame(pos, self.surface)

        pygame.draw.circle(self.surface, fill_color.as_int(), p, round(radius), 0)
        pygame.draw.circle(self.surface, fill_color.as_int(), p, round(radius - 4), 0)

    def draw_segment(self, a: Vec2d, b: Vec2d, color: SpaceDebugColor) -> None:
        p1 = to_pygame(a, self.surface)
        p2 = to_pygame(b, self.surface)

        pygame.draw.aalines(self.surface, color.as_int(), False, [p1, p2])

    def draw_fat_segment(
        self,
        a: Tuple[float, float],
        b: Tuple[float, float],
        radius: float,
        outline_color: SpaceDebugColor,
        fill_color: SpaceDebugColor,
    ) -> None:
        p1 = to_pygame(a, self.surface)
        p2 = to_pygame(b, self.surface)

        r = round(max(1, radius * 2))
        pygame.draw.lines(self.surface, fill_color.as_int(), False, [p1, p2], r)
        if r > 2:
            orthog = [abs(p2[1] - p1[1]), abs(p2[0] - p1[0])]
            if orthog[0] == 0 and orthog[1] == 0:
                return
            scale = radius / (orthog[0] * orthog[0] + orthog[1] * orthog[1]) ** 0.5
            orthog[0] = round(orthog[0] * scale)
            orthog[1] = round(orthog[1] * scale)
            points = [
                (p1[0] - orthog[0], p1[1] - orthog[1]),
                (p1[0] + orthog[0], p1[1] + orthog[1]),
                (p2[0] + orthog[0], p2[1] + orthog[1]),
                (p2[0] - orthog[0], p2[1] - orthog[1]),
            ]
            pygame.draw.polygon(self.surface, fill_color.as_int(), points)
            pygame.draw.circle(
                self.surface,
                fill_color.as_int(),
                (round(p1[0]), round(p1[1])),
                round(radius),
            )
            pygame.draw.circle(
                self.surface,
                fill_color.as_int(),
                (round(p2[0]), round(p2[1])),
                round(radius),
            )

    def draw_polygon(
        self,
        verts: Sequence[Tuple[float, float]],
        radius: float,
        outline_color: SpaceDebugColor,
        fill_color: SpaceDebugColor,
    ) -> None:
        ps = [to_pygame(v, self.surface) for v in verts]
        ps += [ps[0]]

        radius = 2
        pygame.draw.polygon(self.surface, fill_color.as_int(), ps)

        if radius > 0:
            for i in range(len(verts)):
                a = verts[i]
                b = verts[(i + 1) % len(verts)]
                self.draw_fat_segment(a, b, radius, fill_color, fill_color)

    def draw_dot(
        self, size: float, pos: Tuple[float, float], color: SpaceDebugColor
    ) -> None:
        p = to_pygame(pos, self.surface)
        pygame.draw.circle(self.surface, color.as_int(), p, round(size), 0)


def get_mouse_pos(surface: pygame.Surface) -> Tuple[int, int]:
    """Get position of the mouse pointer in pymunk coordinates."""
    p = pygame.mouse.get_pos()
    return from_pygame(p, surface)


def to_pygame(p: Tuple[float, float], surface: pygame.Surface) -> Tuple[int, int]:
    """Convenience method to convert pymunk coordinates to pygame surface
    local coordinates.

    Note that in case positive_y_is_up is False, this function won't actually do
    anything except converting the point to integers.
    """
    if positive_y_is_up:
        return round(p[0]), surface.get_height() - round(p[1])
    else:
        return round(p[0]), round(p[1])


def from_pygame(p: Tuple[float, float], surface: pygame.Surface) -> Tuple[int, int]:
    """Convenience method to convert pygame surface local coordinates to
    pymunk coordinates
    """
    return to_pygame(p, surface)


def light_color(color: SpaceDebugColor):
    color = np.minimum(
        1.2 * np.float32([color.r, color.g, color.b, color.a]), np.float32([255])
    )
    color = SpaceDebugColor(r=color[0], g=color[1], b=color[2], a=color[3])
    return color


def pymunk_to_shapely(body, shapes):
    geoms = list()
    for shape in shapes:
        if isinstance(shape, pymunk.shapes.Poly):
            verts = [body.local_to_world(v) for v in shape.get_vertices()]
            verts += [verts[0]]
            geoms.append(sg.Polygon(verts))
        else:
            raise RuntimeError(f"Unsupported shape type {type(shape)}")
    geom = sg.MultiPolygon(geoms)
    return geom


class PushTetrisEnv(gym.Env):
    """
    Push object environment modified from the Diffusion Policy PushT environment.
    It contains a point agent and a variable number of objects in different shapes.
    The goal is to push the objects to target configurations.
    Reward is calculated based on per-object pixel coverage directly from images using bitmasks for efficiency.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "video_frames_per_second": 10}
    reward_range = (0.0, 1.0)

    def __init__(
        self,
        ob_type='states',
        reward_mode='step',
        mode='task',
        object_list=["O", "I", "S", "Z", "L", "J", "T"],  # shapes: ["O", "I", "S", "Z", "L", "J", "T", "large_T", "+"], original shape is "large_T"
        num_objects=3,
        legacy=False,
        block_cog=None,
        damping=None,
        render_action=False,
        render_size=128,
        reset_to_state=None,
        action_scale=100,
        visualize_target=False,
    ):
        
        self._seed = None
        self.set_seed()

        self.window_size = ws = 512  # The size of the PyGame window
        self.render_size = render_size
        self.sim_hz = 100
        # Local controller params.
        self.k_p, self.k_v = 100, 20  # PD control.z
        self.control_hz = self.metadata["video_frames_per_second"]
        # legcay set_state for data compatibility
        self.legacy = legacy
        self.action_scale = action_scale

        self.shape_to_idx = {"O": 0, "I": 1, "S": 2, "Z": 3, "L": 4, "J": 5, "T": 6, "large_T": 7, "+": 8}
        self.shape_colors = np.array([
            [255, 255, 0],    # Yellow
            [0, 255, 255],    # Cyan
            [255, 0, 0],      # Red
            [0, 255, 0],      # Lime
            [0, 0, 255],      # Blue
            [255, 165, 0],    # Orange
            [160, 32, 240],   # Purple
        ], dtype=np.uint8)  # (N, 3)

        self.object_list = object_list
        self.num_objects = num_objects
        
        assert mode in ['task', 'data_collection']
        self._mode = mode

        assert ob_type in ['states', 'pixels']
        self._ob_type = ob_type

        if self._ob_type == 'pixels':
            # observation space is RGB pixels
            self.observation_space = gym.spaces.Box(
                low=0, high=255, shape=(render_size, render_size, 3), dtype=np.uint8
            )
        else: # 'states'
            # agent_pos, then for each block: block_pos, block_angle
            obs_dim = 2 + 3 * self.num_objects
            low = np.array([0, 0] + [0, 0, 0] * self.num_objects, dtype=np.float32)
            high = np.array([ws, ws] + [ws, ws, 2 * np.pi] * self.num_objects, dtype=np.float32)
            self.observation_space = gym.spaces.Box(
                low=low,
                high=high,
                shape=(obs_dim,),
                dtype=np.float32,
            )

        # relative positional goal for agent
        self.action_space = gym.spaces.Box(
            low=np.array([-1, -1], dtype=np.float64),
            high=np.array([1, 1], dtype=np.float64),
            shape=(2,),
            dtype=np.float64,
        )
        
        self._reward_mode = reward_mode
        self.success_threshold = 0.85  # NOTE: original is 95% coverage

        self.visualize_target = visualize_target
        self.reset_to_state = reset_to_state
        self.coverage_arr = []
        self.chamfer_arr = []
        
        # NOTE: unused legacy parameters below
        self.block_cog = block_cog
        self.damping = damping
        self.render_action = render_action

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None
        self.screen = None

        self.space = None
        self.teleop = None
        self.render_buffer = None
        self.latest_action = None

    def reset(self, seed=None, options=None):
        self._setup()
        if self.block_cog is not None:
            self.block.center_of_gravity = self.block_cog
        if self.damping is not None:
            self.space.damping = self.damping

        # First, force set the current scene to a goal state to obtain the goal observation
        _goal_state = self._sample_state()
        self._set_state(_goal_state)
        self.goal_pose = _goal_state[2:]
        
        goal_state = self._get_state_obs()
        goal_proprio = goal_state[:2]
        goal_visual = self._render_frame("rgb_array")
        self.goal_info = {
            "proprio": goal_proprio,
            "state": goal_state,
            "image": goal_visual,
            "image_bitmasks": self.image_to_reward_masks(goal_visual),
        }
        self.goal_observation = goal_visual if self._ob_type == 'pixels' else goal_state
        
        # use legacy RandomState for compatibility
        _state = self.reset_to_state
        if _state is None:
            _state = self._sample_state()
        self._set_state(_state)

        self.coverage_arr = []
        self.chamfer_arr = []
        state = self._get_state_obs()
        proprio = state[:2]
        visual = self._render_frame("rgb_array")
        info = {
            "proprio": proprio,
            "state": state,
            "image": visual,
            "image_bitmasks": self.image_to_reward_masks(visual),
            "goal": self.goal_observation,
            "goal_rendered": goal_visual,
            "obj_type_idxs": self.cur_obj_indices,
        }
        observation = visual if self._ob_type == 'pixels' else state
        return observation, info

    def step(self, action):
        # pre-step
        prev_state = self._get_state_obs()
        if self._mode == "data_collection":
            prev_image = self._render_frame("rgb_array")
            prev_image_bitmasks = self.image_to_reward_masks(prev_image)
        dt = 1.0 / self.sim_hz
        self.n_contact_points = 0
        n_steps = self.sim_hz // self.control_hz

        # execute action
        if action is not None:
            action = np.array(action) * self.action_scale
            action = self.agent.position + action
            action = np.clip(action, 0, self.window_size)  # positional goal for agent, must be in window
            self.latest_action = action
            for i in range(n_steps):
                # Step PD control.
                # self.agent.velocity = self.k_p * (act - self.agent.position)    # P control works too.
                acceleration = self.k_p * (action - self.agent.position) + self.k_v * (Vec2d(0, 0) - self.agent.velocity)
                self.agent.velocity += acceleration * dt

                # Step physics.
                self.space.step(dt)

        state = self._get_state_obs()
        visual = self._render_frame("rgb_array")
        proprio = state[:2]
        image_bitmasks = self.image_to_reward_masks(visual)
        observation = visual if self._ob_type == 'pixels' else state

        if self._mode == "data_collection":
            reward = 0.0  # no reward needed in data collection mode, saves time
            self.coverage_arr.append(0.0)  # no coverage in data collection mode
            self.chamfer_arr.append(0.0)  # no chamfer in data collection mode
            self.per_obj_coverage = 0.0
        else:
            reward = self.compute_reward(image_bitmasks, self.goal_info["image_bitmasks"])
            self.chamfer_arr.append(self.compute_chamfer_distance(visual, self.goal_info["image"]))
        terminated = False
        truncated = False

        info = self._get_info()
        info["proprio"] = proprio
        info["state"] = state
        info["image"] = visual
        info["image_bitmasks"] = image_bitmasks
        info["obj_type_idxs"] = self.cur_obj_indices
        info["max_coverage"] = max(self.coverage_arr)
        info["final_coverage"] = self.coverage_arr[-1]
        info["min_chamfer"] = min(self.chamfer_arr)
        info["final_chamfer"] = self.chamfer_arr[-1]
        info["success"] = float(np.all(self.per_obj_coverage >= self.success_threshold))
        info["success_frac"] = np.mean(self.per_obj_coverage >= self.success_threshold)
        info["was_agent_object_interaction"] = self._was_agent_object_interaction(state, prev_state)

        if self._mode == "data_collection":
            info["prev_state"] = prev_state
            info["prev_image_bitmasks"] = prev_image_bitmasks
            self._step += 1
            if self._step % 100 == 0:
                self._target_block_id = np.random.randint(0, self.num_objects)

        return observation, reward, terminated, truncated, info
   
    def compute_chamfer_distance(self, image1: np.ndarray, image2: np.ndarray):
        """
        Compute the mean Chamfer distance between per-object pixels in two images,
        only for objects present in the current scene (self.cur_obj_indices).
        If an object is missing in either image, set its Chamfer distance to a high value
        proportional to the grid diagonal.
        Returns:
            chamfer_mean: float, mean Chamfer distance across all present objects
            chamfer_per_object: np.ndarray, shape (num_present,), Chamfer distance per present object
        """
        shape_colors = self.shape_colors  # (N, 3)
        present_indices = np.asarray(self.cur_obj_indices)
        present_colors = shape_colors[present_indices]  # (num_present, 3)
        H, W = image1.shape[:2]
        num_present = len(present_indices)
        chamfer_per_object = np.empty(num_present, dtype=np.float32)

        # Precompute all pixel coordinates
        coords_grid = np.stack(np.meshgrid(np.arange(H), np.arange(W), indexing='ij'), axis=-1)  # (H, W, 2)
        high_value = np.sqrt(H**2 + W**2)

        # Vectorized mask extraction for all present colors
        masks1 = np.all(image1[None, :, :, :] == present_colors[:, None, None, :], axis=-1)  # (num_present, H, W)
        masks2 = np.all(image2[None, :, :, :] == present_colors[:, None, None, :], axis=-1)  # (num_present, H, W)

        for j in range(num_present):
            coords1 = coords_grid[masks1[j]]  # (num_pixels1, 2)
            coords2 = coords_grid[masks2[j]]  # (num_pixels2, 2)
            if coords1.shape[0] == 0 or coords2.shape[0] == 0:
                chamfer_per_object[j] = high_value
                continue
            dists_1_to_2 = cdist(coords1, coords2)
            dists_2_to_1 = cdist(coords2, coords1)
            chamfer = (np.mean(np.min(dists_1_to_2, axis=1)) + np.mean(np.min(dists_2_to_1, axis=1))) / 2.0
            chamfer_per_object[j] = chamfer

        chamfer_mean = chamfer_per_object.mean()
        normalized_chamfer_mean = chamfer_mean / 41  # NOTE: 41 is the empirical chamfer distance between randomly initialized start and goal states for 3 objects
        return normalized_chamfer_mean
    
    def image_to_reward_masks(self, image):
        # Create masks for each color
        masks = np.all(image[None, :, :, :] == self.shape_colors[:, None, None, :], axis=-1)  # (N, H, W)
        packed = np.packbits(masks.reshape(self.shape_colors.shape[0], -1), axis=-1)
        packed = packed[self.cur_obj_indices]
        return packed
    
    def compute_reward(self, obs_masks_packed, goal_masks_packed):
        """ Compute reward based on packed color masks of the observation and goal images."""

        # Create a lookup table for popcount (number of set bits in uint8)
        popcount_lut = np.array(_POPCOUNT_TABLE8, dtype=np.uint8)

        # Compute area and intersection per object
        area_goal = popcount_lut[goal_masks_packed].sum(axis=-1)
        intersection = popcount_lut[(obs_masks_packed & goal_masks_packed)].sum(axis=-1)

        # Coverage: intersection / area_goal (avoid div by zero)
        coverage = intersection / np.maximum(area_goal, 1)  # avoid div by zero

        # Reward calculation
        if self._reward_mode == 'sparse':
            reward = 0.0 if np.all(coverage >= self.success_threshold) else -1.0
        elif self._reward_mode == 'step':
            reward = -np.mean(coverage < self.success_threshold)
        else:  # 'dense'
            reward = np.mean(coverage) - 1.0

        # Optionally store for info/debug
        self.coverage_arr.append(np.mean(coverage))
        self.per_obj_coverage = coverage

        return reward
    
    def compute_reward_batch(self, obs_masks_packed: torch.Tensor, goal_masks_packed: torch.Tensor):
        """
        Compute rewards for a batch of bitpacked masks (torch tensors), considering only shapes present in each sample.
        Args:
            obs_masks_packed: (B, N, packed_len) torch.uint8
            goal_masks_packed: (B, N, packed_len) torch.uint8
        Returns:
            rewards: (B,) torch.float32
        """
        # Create a lookup table for popcount (number of set bits in uint8)
        popcount_lut = torch.tensor(_POPCOUNT_TABLE8, dtype=torch.uint8, device=obs_masks_packed.device)

        # Count bits for each mask
        # Shape: (B, N, packed_len) -> (B, N)
        area_goal = popcount_lut[goal_masks_packed.to(torch.long)].sum(dim=-1)
        intersection = popcount_lut[(obs_masks_packed & goal_masks_packed).to(torch.long)].sum(dim=-1)

        # Coverage: intersection / area_goal (avoid div by zero)
        coverage = intersection.float() / area_goal.float().clamp(min=1)

        # Vectorized reward calculation
        if self._reward_mode == 'sparse':
            # Success if all present shapes are above threshold
            success = (coverage >= self.success_threshold)
            reward = torch.where(success.all(dim=1), 0.0, -1.0)
        elif self._reward_mode == 'step':
            reward = -torch.mean((coverage < self.success_threshold).float(), dim=1)
        else:  # 'dense'
            reward = torch.mean(coverage, dim=1) - 1.0

        return reward
    
    def render(self, mode="rgb_array"):
        return self._render_frame(mode)

    def get_data_collection_action(self):
        assert self._mode == "data_collection"
        # sample a random action
        action = np.random.uniform(-1, 1, size=self.action_space.shape)
        # scale action to environment position
        action = action * self.action_scale
        action = self.agent.position + action
        # constrain agent to be near a target block to increase probability of interaction
        target_block_pos = np.array((self.blocks[self._target_block_id].position.x, self.blocks[self._target_block_id].position.y))
        action = np.clip(action, target_block_pos - 100, target_block_pos + 100)
        # rescale action back to action space
        action = (action - self.agent.position) / self.action_scale
        action = np.clip(action, self.action_space.low, self.action_space.high)
        return action
    
    def teleop_agent(self):
        TeleopAgent = collections.namedtuple("TeleopAgent", ["act"])

        def act(obs):
            act = None
            mouse_position = pymunk.pygame_util.from_pygame(Vec2d(*pygame.mouse.get_pos()), self.screen)
            if self.teleop or (mouse_position - self.agent.position).length < 30:
                self.teleop = True
                act = mouse_position
            return act

        return TeleopAgent(act)

    def _sample_state(self):
        rs = self.random_state
        patch_size = 128
        grid_size = self.window_size // patch_size  # 4

        # Randomly select internal patch corners for agent and objects position without overlap
        internal_corners = [
            (i * patch_size, j * patch_size)
            for i in range(1, grid_size)
            for j in range(1, grid_size)
        ]

        num_entities = self.num_objects + 1  # agent + objects
        chosen_indices = rs.choice(len(internal_corners), size=num_entities, replace=False)
        chosen_corners = [internal_corners[idx] for idx in chosen_indices]
        
        # Agent position (no orientation)
        agent_pos = chosen_corners[0]

        # Object positions and orientations
        obj_states = [
            val
            for pos in chosen_corners[1:]
            for val in (pos[0], pos[1], rs.uniform(0, 2 * np.pi))
        ]

        # State: [agent_x, agent_y, obj1_x, obj1_y, obj1_theta, ...]
        state = np.array(list(agent_pos) + obj_states, dtype=np.float32)
        return state

    def _get_state_obs(self):
        # Agent position and angle (always 0), plus 1-hot for "agent"
        obs = list(self.agent.position) + [0.0]  # angle = 0
        agent_onehot = [0.0] * (len(self.object_list) + 1)
        agent_onehot[0] = 1.0
        obs.extend(agent_onehot)

        # For each block, append (x, y, theta) and 1-hot for its type (global object list)
        for i, block in enumerate(self.blocks):
            obs.extend([block.position.x, block.position.y, block.angle])
            onehot = [0.0] * (len(self.object_list) + 1)
            idx = self.cur_obj_indices[i] + 1
            onehot[idx] = 1.0
            obs.extend(onehot)
        return np.array(obs, dtype=np.float32)
    
    def _get_goal_pose_body(self, pose):
        mass = 1
        inertia = pymunk.moment_for_box(mass, (50, 100))
        body = pymunk.Body(mass, inertia)
        # preserving the legacy assignment order for compatibility
        # the order here doesn't matter somehow, maybe because CoM is aligned with body origin
        body.position = pose[:2].tolist()
        body.angle = pose[2]
        return body

    def _get_info(self):
        n_steps = self.sim_hz // self.control_hz
        n_contact_points_per_step = int(np.ceil(self.n_contact_points / n_steps))
        info = {
            "pos_agent": np.array(self.agent.position),
            "vel_agent": np.array(self.agent.velocity),
            "block_pose": np.array(
                [[block.position.x, block.position.y, block.angle] for block in self.blocks]
            ),
            "goal_pose": self.goal_pose,
            "n_contacts": n_contact_points_per_step,
        }
        return info

    def _render_frame(self, mode):
        if self.window is None and mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        self.screen = canvas

        draw_options = DrawOptions(canvas)

        # Draw goal pose.
        for i, block in enumerate(self.blocks):
            if self._mode == "data_collection" and i != self._target_block_id:
                continue
            goal_pose = self.goal_pose[3 * i : 3 * (i + 1)]
            goal_body = self._get_goal_pose_body(goal_pose)
            for shape in block.shapes:
                goal_points = [
                    pymunk.pygame_util.to_pygame(goal_body.local_to_world(v), draw_options.surface)
                    for v in shape.get_vertices()
                ]
                goal_points += [goal_points[0]]
                pygame.draw.polygon(canvas, self.goal_color, goal_points)

        # Draw agent and blocks.
        self.space.debug_draw(draw_options)

        if mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            # the clock is already ticked during in step for "human"

        img = np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))
        if self.render_size != self.window_size:
            img = cv2.resize(img, (self.render_size, self.render_size), interpolation=cv2.INTER_NEAREST)  # NOTE: using INTER_NEAREST to preserve pixel values for image-based reward computation
        if self.render_action:
            if self.render_action and (self.latest_action is not None):
                action = np.array(self.latest_action)
                coord = (action / 512 * 96).astype(np.int32)
                marker_size = int(8 / 96 * self.render_size)
                thickness = int(1 / 96 * self.render_size)
                cv2.drawMarker(
                    img,
                    coord,
                    color=(255, 0, 0),
                    markerType=cv2.MARKER_CROSS,
                    markerSize=marker_size,
                    thickness=thickness,
                )
        return img

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def set_seed(self, seed=None):
        if seed is None:
            seed = np.random.randint(0, 25536)
        self._seed = seed
        self.np_random = np.random.default_rng(seed)
        self.random_state = np.random.RandomState(seed)

    def _handle_collision(self, arbiter, space, data):
        self.n_contact_points += len(arbiter.contact_point_set.points)

    def _set_state(self, state):
        if isinstance(state, np.ndarray):
            state = state.tolist()
        pos_agent = state[:2]
        self.agent.velocity = (0, 0)
        self.agent.position = pos_agent

        # Set state for all blocks
        block_state = state[2:]
        for i, block in enumerate(self.blocks):
            idx = i * 3
            pos_block = block_state[idx:idx+2]
            rot_block = block_state[idx+2]
            # setting angle rotates with respect to center of mass
            # therefore will modify the geometric position
            # if not the same as CoM
            # therefore should be modified first.
            if self.legacy:
                block.position = pos_block
                block.angle = rot_block
            else:
                block.angle = rot_block
                block.position = pos_block

        # Run physics to take effect
        self.space.step(1.0 / self.sim_hz)

    def _was_agent_object_interaction(self, state, prev_state):
        object_xys = state.reshape(self.num_objects + 1, -1)[1:, :2]
        prev_object_xys = prev_state.reshape(self.num_objects + 1, -1)[1:, :2]

        dists = np.linalg.norm(prev_object_xys - object_xys, ord=1, axis=-1)
        return np.any(dists >= 3, axis=-1)
    
    def _setup(self):
        self.space = pymunk.Space()
        self.space.gravity = 0, 0
        self.space.damping = 0
        self.teleop = False
        self.render_buffer = list()

        # Add walls
        walls = [
            self._add_segment((5, 506), (5, 5), 2),
            self._add_segment((5, 5), (506, 5), 2),
            self._add_segment((506, 5), (506, 506), 2),
            self._add_segment((5, 506), (506, 506), 2),
        ]
        self.space.add(*walls)
        
        # Add agent
        self.agent = self.add_circle((256, 400), 15)
        
        # Add randomized blocks for this episode
        self.cur_obj_indices = self.random_state.choice(len(self.object_list), size=self.num_objects, replace=False)
        self.cur_object_list = [self.object_list[i] for i in self.cur_obj_indices]
        self.blocks = []
        for i, shape in enumerate(self.cur_object_list):
            self.blocks.append(self.add_shape(shape, ((i // 4) * 64 + 32, (i % 4) * 64 + 32), 0, scale=30))  # default scale is 40
        
        # Add collision handling
        self.collision_handeler = self.space.add_collision_handler(0, 0)
        self.collision_handeler.post_solve = self._handle_collision
        self.n_contact_points = 0

        if self.visualize_target:
            self.goal_color = pygame.Color("LightSlateGray")
        else:
            self.goal_color = pygame.Color("White")
        
        # Data collection variables
        self._step = 0
        self._target_block_id = np.random.randint(0, self.num_objects)

    def _add_segment(self, a, b, radius):
        shape = pymunk.Segment(self.space.static_body, a, b, radius)
        shape.color = pygame.Color("LightGray")  # https://htmlcolorcodes.com/color-names
        return shape

    def add_circle(self, position, radius):
        body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        body.position = position
        body.friction = 1
        shape = pymunk.Circle(body, radius)
        # shape.color = pygame.Color("RoyalBlue")
        shape.color = pygame.Color("Black")
        self.space.add(body, shape)
        return body

    def add_box(self, position, height, width):
        mass = 1
        inertia = pymunk.moment_for_box(mass, (height, width))
        body = pymunk.Body(mass, inertia)
        body.position = position
        shape = pymunk.Poly.create_box(body, (height, width))
        shape.color = pygame.Color("LightSlateGray")
        self.space.add(body, shape)
        return body

    def add_T(
        self,
        position,
        angle,
        scale=30,
        color="Purple",
        mask=pymunk.ShapeFilter.ALL_MASKS(),
    ):
        mass = 1
        vertices1 = [
            (-3 * scale / 2, scale),
            (3 * scale / 2, scale),
            (3 * scale / 2, 0),
            (-3 * scale / 2, 0),
        ]
        inertia1 = pymunk.moment_for_poly(mass, vertices=vertices1)
        vertices2 = [
            (-scale / 2, scale),
            (-scale / 2, 2 * scale),
            (scale / 2, 2 * scale),
            (scale / 2, scale),
        ]
        inertia2 = pymunk.moment_for_poly(mass, vertices=vertices2)
        body = pymunk.Body(mass, inertia1 + inertia2)
        shape1 = pymunk.Poly(body, vertices1)
        shape2 = pymunk.Poly(body, vertices2)
        shape1.color = pygame.Color(color)
        shape2.color = pygame.Color(color)
        shape1.filter = pymunk.ShapeFilter(mask=mask)
        shape2.filter = pymunk.ShapeFilter(mask=mask)
        body.center_of_gravity = (shape1.center_of_gravity + shape2.center_of_gravity) / 2
        body.position = position
        body.angle = angle
        body.friction = 1
        self.space.add(body, shape1, shape2)
        return body
    
    def add_O(
        self,
        position,
        angle,
        scale=30,
        color="Yellow",
        mask=pymunk.ShapeFilter.ALL_MASKS(),
    ):
        mass = 1
        vertices1 = [
            (-scale, -scale),
            (-scale, scale),
            (scale, scale),
            (scale, -scale),
        ]
        inertia1 = pymunk.moment_for_poly(mass, vertices=vertices1)
        body = pymunk.Body(mass, inertia1)
        shape1 = pymunk.Poly(body, vertices1)
        shape1.color = pygame.Color(color)
        shape1.filter = pymunk.ShapeFilter(mask=mask)
        body.center_of_gravity = shape1.center_of_gravity
        body.position = position
        body.angle = angle
        body.friction = 1
        self.space.add(body, shape1)
        return body
    
    def add_I(
        self,
        position,
        angle,
        scale=30,
        color="Cyan",
        mask=pymunk.ShapeFilter.ALL_MASKS(),
    ):
        mass = 1
        vertices1 = [
            (-scale / 2, -scale * 2),
            (-scale / 2, scale * 2),
            (scale / 2, scale * 2),
            (scale / 2, -scale * 2),
        ]
        inertia1 = pymunk.moment_for_poly(mass, vertices=vertices1)
        body = pymunk.Body(mass, inertia1)
        shape1 = pymunk.Poly(body, vertices1)
        shape1.color = pygame.Color(color)
        shape1.filter = pymunk.ShapeFilter(mask=mask)
        body.center_of_gravity = shape1.center_of_gravity
        body.position = position
        body.angle = angle
        body.friction = 1
        self.space.add(body, shape1)
        return body
    
    def add_S(
        self,
        position,
        angle,
        scale=30,
        color="Red",
        mask=pymunk.ShapeFilter.ALL_MASKS(),
    ):
        # S shape: like Z but mirrored
        mass = 1
        length = 2
        # Lower rectangle (right)
        vertices1 = [
            (0, 0),
            (0, length * scale / 2),
            (-length * scale, length * scale / 2),
            (-length * scale, 0),
        ]
        inertia1 = pymunk.moment_for_poly(mass, vertices=vertices1)
        # Upper rectangle (left)
        vertices2 = [
            (length * scale / 2, 0),
            (length * scale / 2, -length * scale / 2),
            (-length * scale / 2, -length * scale / 2),
            (-length * scale / 2, 0),
        ]
        inertia2 = pymunk.moment_for_poly(mass, vertices=vertices2)
        body = pymunk.Body(mass, inertia1 + inertia2)
        shape1 = pymunk.Poly(body, vertices1)
        shape2 = pymunk.Poly(body, vertices2)
        shape1.color = pygame.Color(color)
        shape2.color = pygame.Color(color)
        shape1.filter = pymunk.ShapeFilter(mask=mask)
        shape2.filter = pymunk.ShapeFilter(mask=mask)
        body.center_of_gravity = (shape1.center_of_gravity + shape2.center_of_gravity) / 2
        body.position = position
        body.angle = angle
        body.friction = 1
        self.space.add(body, shape1, shape2)
        return body

    def add_Z(
        self,
        position,
        angle,
        scale=30,
        color="Lime",
        mask=pymunk.ShapeFilter.ALL_MASKS(),
    ):
        mass = 1
        length = 2
        vertices1 = [
            (0, 0),
            (0, length * scale / 2),
            (length * scale, length * scale / 2),
            (length * scale, 0),
        ]
        inertia1 = pymunk.moment_for_poly(mass, vertices=vertices1)
        vertices2 = [
            (-length * scale / 2, 0),
            (length * scale / 2, 0),
            (length * scale / 2, -length * scale / 2),
            (-length * scale / 2, -length * scale / 2),
        ]
        inertia2 = pymunk.moment_for_poly(mass, vertices=vertices2)
        body = pymunk.Body(mass, inertia1 + inertia2)
        shape1 = pymunk.Poly(body, vertices1)
        shape2 = pymunk.Poly(body, vertices2)
        shape1.color = pygame.Color(color)
        shape2.color = pygame.Color(color)
        shape1.filter = pymunk.ShapeFilter(mask=mask)
        shape2.filter = pymunk.ShapeFilter(mask=mask)
        body.center_of_gravity = (
            shape1.center_of_gravity + shape2.center_of_gravity
        ) / 2
        body.position = position
        body.angle = angle
        body.friction = 1
        self.space.add(body, shape1, shape2)
        return body

    def add_J(
        self,
        position,
        angle,
        scale=30,
        color="Orange",
        mask=pymunk.ShapeFilter.ALL_MASKS(),
    ):
        mass = 1
        length = 2
        vertices1 = [
            (0, 0),
            (0, scale * length),
            (scale * length / 2, scale * length),
            (scale * length / 2, 0),
        ]
        inertia1 = pymunk.moment_for_poly(mass, vertices=vertices1)
        vertices2 = [
            (0, 0),
            (scale * length, 0),
            (scale * length, -scale * length / 2),
            (0, -scale * length / 2),
        ]
        inertia2 = pymunk.moment_for_poly(mass, vertices=vertices2)
        body = pymunk.Body(mass, inertia1 + inertia2)
        shape1 = pymunk.Poly(body, vertices1)
        shape2 = pymunk.Poly(body, vertices2)
        shape1.color = pygame.Color(color)
        shape2.color = pygame.Color(color)
        shape1.filter = pymunk.ShapeFilter(mask=mask)
        shape2.filter = pymunk.ShapeFilter(mask=mask)
        body.center_of_gravity = (
            shape1.center_of_gravity + shape2.center_of_gravity
        ) / 2
        body.position = position
        body.angle = angle
        body.friction = 1
        self.space.add(body, shape1, shape2)
        return body

    def add_L(
        self,
        position,
        angle,
        scale=30,
        color="Blue",
        mask=pymunk.ShapeFilter.ALL_MASKS(),
    ):
        mass = 1
        length = 2
        # Vertical bar on the right
        vertices1 = [
            (0, 0),
            (0, scale * length),
            (-scale * length / 2, scale * length),
            (-scale * length / 2, 0),
        ]
        inertia1 = pymunk.moment_for_poly(mass, vertices=vertices1)
        # Horizontal foot to the left
        vertices2 = [
            (0, 0),
            (-scale * length, 0),
            (-scale * length, -scale * length / 2),
            (0, -scale * length / 2),
        ]
        inertia2 = pymunk.moment_for_poly(mass, vertices=vertices2)
        body = pymunk.Body(mass, inertia1 + inertia2)
        shape1 = pymunk.Poly(body, vertices1)
        shape2 = pymunk.Poly(body, vertices2)
        shape1.color = pygame.Color(color)
        shape2.color = pygame.Color(color)
        shape1.filter = pymunk.ShapeFilter(mask=mask)
        shape2.filter = pymunk.ShapeFilter(mask=mask)
        body.center_of_gravity = (shape1.center_of_gravity + shape2.center_of_gravity) / 2
        body.position = position
        body.angle = angle
        body.friction = 1
        self.space.add(body, shape1, shape2)
        return body

    def add_large_T(
        self,
        position,
        angle,
        scale=30,
        color="Red",
        mask=pymunk.ShapeFilter.ALL_MASKS(),
    ):  
        mass = 1
        length = 4
        vertices1 = [
            (-length * scale / 2, scale),
            (length * scale / 2, scale),
            (length * scale / 2, 0),
            (-length * scale / 2, 0),
        ]
        inertia1 = pymunk.moment_for_poly(mass, vertices=vertices1)
        vertices2 = [
            (-scale / 2, scale),
            (-scale / 2, length * scale),
            (scale / 2, length * scale),
            (scale / 2, scale),
        ]
        inertia2 = pymunk.moment_for_poly(mass, vertices=vertices1)
        body = pymunk.Body(mass, inertia1 + inertia2)
        shape1 = pymunk.Poly(body, vertices1)
        shape2 = pymunk.Poly(body, vertices2)
        shape1.color = pygame.Color(color)
        shape2.color = pygame.Color(color)
        shape1.filter = pymunk.ShapeFilter(mask=mask)
        shape2.filter = pymunk.ShapeFilter(mask=mask)
        body.center_of_gravity = (shape1.center_of_gravity + shape2.center_of_gravity) / 2
        body.position = position
        body.angle = angle
        body.friction = 1
        self.space.add(body, shape1, shape2)
        return body
    
    def add_plus(
        self,
        position,
        angle,
        scale=30,
        color="Blue",
        mask=pymunk.ShapeFilter.ALL_MASKS(),
    ):
        mass = 1
        vertices1 = [
            (-3 * scale / 2, scale / 2),
            (3 * scale / 2, scale / 2),
            (3 * scale / 2, -scale / 2),
            (-3 * scale / 2, -scale / 2),
        ]
        inertia1 = pymunk.moment_for_poly(mass, vertices=vertices1)
        vertices2 = [
            (-scale / 2, scale / 2),
            (-scale / 2, 3 * scale / 2),
            (scale / 2, scale / 2),
            (scale / 2, 3 * scale / 2),
        ]
        vertices3 = [
            (-scale / 2, -scale / 2),
            (-scale / 2, -3 * scale / 2),
            (scale / 2, -scale / 2),
            (scale / 2, -3 * scale / 2),
        ]
        inertia2 = pymunk.moment_for_poly(mass, vertices=vertices1)
        inertia3 = pymunk.moment_for_poly(mass, vertices=vertices1)
        body = pymunk.Body(mass, inertia1 + inertia2 + inertia3)
        shape1 = pymunk.Poly(body, vertices1)
        shape2 = pymunk.Poly(body, vertices2)
        shape3 = pymunk.Poly(body, vertices3)
        shape1.color = pygame.Color(color)
        shape2.color = pygame.Color(color)
        shape3.color = pygame.Color(color)
        shape1.filter = pymunk.ShapeFilter(mask=mask)
        shape2.filter = pymunk.ShapeFilter(mask=mask)
        shape3.filter = pymunk.ShapeFilter(mask=mask)
        body.center_of_gravity = (
            shape1.center_of_gravity + shape2.center_of_gravity + shape3.center_of_gravity
        ) / 3
        body.position = position
        body.angle = angle
        body.friction = 1
        self.space.add(body, shape1, shape2, shape3)
        return body

    def add_shape(self, shape, *args, **kwargs):
        # Dispatch method based on the 'shape' parameter

        # Tetris shapes
        if shape == "T":
            return self.add_T(*args, **kwargs)
        elif shape == "O":
            return self.add_O(*args, **kwargs)
        elif shape == "I":
            return self.add_I(*args, **kwargs)
        elif shape == "S":
            return self.add_S(*args, **kwargs)
        elif shape == "Z":
            return self.add_Z(*args, **kwargs)
        elif shape == "L":
            return self.add_L(*args, **kwargs)
        elif shape == "J":
            return self.add_J(*args, **kwargs)
        
        # Non-tetris shapes
        elif shape == "large_T":
            return self.add_large_T(*args, **kwargs)
        elif shape == "+":
            return self.add_plus(*args, **kwargs)
        
        else:
            raise ValueError(f"Unknown shape type: {shape}")
