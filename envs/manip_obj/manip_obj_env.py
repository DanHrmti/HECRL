from pathlib import Path

import gymnasium as gym
from gymnasium.spaces import Box
import mujoco
from dm_control import mjcf

import numpy as np
import torch

import ogbench
from ogbench.manipspace import controllers, lie, mjcf_utils
from ogbench.manipspace.envs.env import CustomMuJoCoEnv


class ManipObjEnv(CustomMuJoCoEnv):
    """Compositional object environment modified from the OGBench CubeEnv.

    It contains a UR5e robot arm with a Robotiq 2F-85 gripper.
    The default control mode is relative end-effector control. The 5-D action space corresponds to the following:
    - 3-D relative end-effector position (x, y, z).
    - 1-D relative end-effector yaw.
    - 1-D relative gripper opening.

    This environment consists of a single or multiple objects. The goal is to move the objects to target configurations.
    """

    def __init__(
        self,
        ob_type='states',
        multiview=False,
        mode='task',
        reward_mode='step',
        reward_scale=1.0,
        num_cubes=1,
        num_colors=6,
        random_colors=False,
        terminate_at_goal=False,
        visualize_info=False,
        physics_timestep=0.002,
        control_timestep=0.05,
        **kwargs,
    ):
        """Initialize the environment.

        Args:
            ob_type: Observation type. Either 'states', 'states_clean' or 'pixels'.
                'states' correspond to the full state specified by OGBench.
                'states_clean' correspond to a subset of 'states' that excludes some information that is not necessary to solve the task (e.g., velocities).
            multiview: Whether to use multi-view pixel observations. Only applicable when ob_type is 'pixels'.
            mode: Mode of the environment. Either 'task' or 'data_collection'. In 'task' mode, the environment is used
                for training and evaluation. In 'data_collection' mode, the environment is used for collecting offline
                data. 'task_stack' and 'data_collection_stack' are variants that focus on stacking.
                'task_sort' is a variant for evaluating specific compositional generalization scenarios.
            reward_mode: Either 'sparse', 'step' or 'dense'.
                'sparse': reward of 0 when the task is completed, -1 otherwise.
                'step': stepwise sparse reward for each completed subtask. Reward is -n/N, where n is the number of remaining subtasks and N is the total number of subtasks.
                'dense': dense reward based on the distance between the objects and the target configurations.
            reward_scale: Scale factor for the dense reward.
            num_cubes: Number of cube objects in the environment.
            num_colors: Number of colors to use for the objects.
            random_colors: Whether to use random colors for the objects.
                If so, colors are randomly assigned to objects at the beginning of each episode without repetition (unless mode=='task_sort').
            terminate_at_goal: Whether to terminate the episode when the goal is reached.
            visualize_info: Whether to visualize the task information (e.g., success status).
            physics_timestep: Physics timestep.
            control_timestep: Control timestep.
            **kwargs: Additional keyword arguments.
        """
                
        super().__init__(
            physics_timestep=physics_timestep,
            control_timestep=control_timestep,
            **kwargs,
        )

        # Define constants.
        self._desc_dir = Path(ogbench.manipspace.__file__).resolve().parent / 'descriptions'
        self._home_qpos = np.asarray([-np.pi / 2, -np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, 0])
        self._effector_down_rotation = lie.SO3(np.asarray([0.0, 1.0, 0.0, 0.0]))
        self._workspace_bounds = np.asarray([[0.25, -0.35, 0.02], [0.6, 0.35, 0.35]])
        self._arm_sampling_bounds = np.asarray([[0.25, -0.35, 0.20], [0.6, 0.35, 0.35]])
        self._object_sampling_bounds = np.asarray([[0.3, -0.3], [0.55, 0.3]])
        self._target_sampling_bounds = np.asarray([[0.3, -0.3], [0.55, 0.3]])
        self._sort_goal_positions = np.asarray([[0.3, -0.2], [0.3, 0.2], [0.55, -0.2], [0.55, 0.2], [0.425, 0.0]])
        self._object_colors = np.array(
            [
                [0.96, 0.26, 0.33, 1.0],  # red
                [0.06, 0.74, 0.21, 1.0],  # green
                [0.35, 0.55, 0.91, 1.0],  # blue
                [1.00, 0.69, 0.21, 1.0],  # orange
                [0.61, 0.28, 0.82, 1.0],  # purple
                [0.00, 0.85, 0.85, 1.0],  # cyan / teal
                # [0.90, 0.90, 0.90, 1.0],  # white
                # [0.10, 0.10, 0.10, 1.0],  # black
            ]
        )
        
        self.agent_state_dim = 6 if ob_type == 'states_clean' else 19
        self.object_state_dim = 5 if ob_type == 'states_clean' else 9

        self._ob_type = ob_type
        self._multiview = multiview
        self._mode = mode
        self._reward_mode = reward_mode
        self._reward_scale = reward_scale
        self.num_cubes = num_cubes
        self.num_objects = num_cubes
        self._num_colors = num_colors
        self._random_colors = random_colors
        self._terminate_at_goal = terminate_at_goal
        self._visualize_info = visualize_info
        
        assert ob_type in ['states', 'states_clean', 'pixels']
        assert reward_mode in ['sparse', 'step', 'dense'], f"Invalid reward mode: {reward_mode}."
        assert num_cubes > 0, "At least one object must be present in the environment."
        assert num_colors <= len(self._object_colors), f"Number of colors ({num_colors}) exceeds available colors ({len(self._object_colors)})."

        # Initialize inverse kinematics controller.
        ik_mjcf = mjcf.from_path((self._desc_dir / 'universal_robots_ur5e' / 'ur5e.xml'), escape_separators=True)
        xml_str = mjcf_utils.to_string(ik_mjcf)
        assets = mjcf_utils.get_assets(ik_mjcf)
        ik_model = mujoco.MjModel.from_xml_string(xml_str, assets)

        self._ik = controllers.DiffIKController(model=ik_model, sites=['attachment_site'])

        # Define action space.
        action_range = np.array([0.05, 0.05, 0.05, 0.3, 1.0])
        self.action_low = -action_range
        self.action_high = action_range

        if 'task' in self._mode:
            # Set task goals.
            self._cur_goal_ob = None
            self._cur_goal_rendered = None
            self._render_goal = False
        else:
            # Target info.
            self._target_task = 'push_cube'
            # The target cube position is stored in the mocap object.
            self._target_block = 0
            self._stacking = False

        self._success = False

    @property
    def observation_space(self):
        if self._model is None:
            self.reset()

        ex_ob = self.compute_observation()

        if self._ob_type == 'pixels':
            return Box(low=0, high=255, shape=ex_ob.shape, dtype=ex_ob.dtype)
        else:
            return Box(low=-np.inf, high=np.inf, shape=ex_ob.shape, dtype=ex_ob.dtype)

    @property
    def action_space(self):
        return gym.spaces.Box(
            low=-np.ones(5),
            high=np.ones(5),
            shape=(5,),
            dtype=np.float32,
        )

    def normalize_action(self, action):
        """Normalize the action to the range [-1, 1]."""
        action = 2 * (action - self.action_low) / (self.action_high - self.action_low) - 1
        return np.clip(action, -1, 1)

    def unnormalize_action(self, action):
        """Unnormalize the action to the range [action_low, action_high]."""
        return 0.5 * (action + 1) * (self.action_high - self.action_low) + self.action_low

    def build_mjcf_model(self):
        # Set scene.
        arena_mjcf = mjcf.from_path((self._desc_dir / 'floor_wall.xml').as_posix())
        arena_mjcf.model = 'ur5e_arena'

        arena_mjcf.statistic.center = (0.3, 0, 0.15)
        arena_mjcf.statistic.extent = 0.7
        getattr(arena_mjcf.visual, 'global').elevation = -20
        getattr(arena_mjcf.visual, 'global').azimuth = 180
        arena_mjcf.statistic.meansize = 0.04
        arena_mjcf.visual.map.znear = 0.1
        arena_mjcf.visual.map.zfar = 10.0

        # Add UR5e robot arm.
        ur5e_mjcf = mjcf.from_path((self._desc_dir / 'universal_robots_ur5e' / 'ur5e.xml'), escape_separators=True)
        ur5e_mjcf.model = 'ur5e'

        for light in ur5e_mjcf.find_all('light'):
            light.remove()
            del light

        # Attach the robotiq gripper to the UR5e flange.
        gripper_mjcf = mjcf.from_path((self._desc_dir / 'robotiq_2f85' / '2f85.xml'), escape_separators=True)
        gripper_mjcf.model = 'robotiq'
        mjcf_utils.attach(ur5e_mjcf, gripper_mjcf, 'attachment_site')

        # Attach UR5e to the scene.
        mjcf_utils.attach(arena_mjcf, ur5e_mjcf)

        self.add_objects(arena_mjcf)

        # Cache joint and actuator elements.
        self._arm_jnts = mjcf_utils.safe_find_all(
            ur5e_mjcf,
            'joint',
            exclude_attachments=True,
        )
        self._arm_acts = mjcf_utils.safe_find_all(
            ur5e_mjcf,
            'actuator',
            exclude_attachments=True,
        )
        self._gripper_jnts = mjcf_utils.safe_find_all(gripper_mjcf, 'joint', exclude_attachments=True)
        self._gripper_acts = mjcf_utils.safe_find_all(gripper_mjcf, 'actuator', exclude_attachments=True)

        # Add bounding boxes to visualize the workspace and object sampling bounds.
        mjcf_utils.add_bounding_box_site(
            arena_mjcf.worldbody,
            lower=np.asarray((*self._target_sampling_bounds[0], 0.02)),
            upper=np.asarray((*self._target_sampling_bounds[1], 0.02)),
            rgba=(0.6, 0.3, 0.3, 0.2),
            group=4,
            name='object_bounds',
        )
        mjcf_utils.add_bounding_box_site(
            arena_mjcf.worldbody,
            lower=np.asarray(self._arm_sampling_bounds[0]),
            upper=np.asarray(self._arm_sampling_bounds[1]),
            rgba=(0.3, 0.6, 0.3, 0.2),
            group=4,
            name='arm_bounds',
        )

        return arena_mjcf

    def add_objects(self, arena_mjcf):
        # Add objects.
        self._object_geoms_list = []
        self._object_target_geoms_list = []
        if self.num_cubes > 0:
            self.add_cubes(arena_mjcf)
        
        # Add cameras.
        cameras = {
            'front_pixels': {
                'pos': (1.053, -0.014, 0.639),
                'xyaxes': (0.000, 1.000, 0.000, -0.628, 0.001, 0.778),
            },
            'side_pixels': {
                'pos': (0.414, -0.753, 0.639),
                'xyaxes': (1.000, 0.000, 0.000, -0.001, 0.628, 0.778),
            },
            'side_pixels_stack': {
                'pos': (0.400, -0.987, 0.509),
                'xyaxes': (1.000, 0.000, 0.000, 0.00, 0.342, 0.940),
            },
        }
        for camera_name, camera_kwargs in cameras.items():
            arena_mjcf.worldbody.add('camera', name=camera_name, **camera_kwargs)

    def add_cubes(self, arena_mjcf):
        # Add cube scene.
        cube_outer_mjcf = mjcf.from_path((self._desc_dir / 'cube_outer.xml').as_posix())
        arena_mjcf.include_copy(cube_outer_mjcf)

        # Add `num_cubes` cubes to the scene.
        distance = 0.05
        for i in range(self.num_cubes):
            cube_mjcf = mjcf.from_path((self._desc_dir / 'cube_inner.xml').as_posix())
            pos = -distance * (self.num_cubes - 1) + 2 * distance * i
            cube_mjcf.find('body', 'object_0').pos[1] = pos
            cube_mjcf.find('body', 'object_target_0').pos[1] = pos
            cube_mjcf.find('geom', 'object_0').density = 8000  # NOTE: change default density of OGBench cube (1240), dataset collected with 8000
            for tag in ['body', 'joint', 'geom', 'site']:
                for item in cube_mjcf.find_all(tag):
                    if hasattr(item, 'name') and item.name is not None and item.name.endswith('_0'):
                        item.name = item.name[:-2] + f'_{i}'
            arena_mjcf.include_copy(cube_mjcf)

        # Save cube geoms.
        for i in range(self.num_cubes):
            self._object_geoms_list.append(arena_mjcf.find('body', f'object_{i}').find_all('geom'))
        for i in range(self.num_cubes):
            self._object_target_geoms_list.append(arena_mjcf.find('body', f'object_target_{i}').find_all('geom'))
    
    def post_compilation(self):
        # Arm joint and actuator IDs.
        arm_joint_names = [j.full_identifier for j in self._arm_jnts]
        self._arm_joint_ids = np.asarray([self._model.joint(name).id for name in arm_joint_names])
        actuator_names = [a.full_identifier for a in self._arm_acts]
        self._arm_actuator_ids = np.asarray([self._model.actuator(name).id for name in actuator_names])
        gripper_actuator_names = [a.full_identifier for a in self._gripper_acts]
        self._gripper_actuator_ids = np.asarray([self._model.actuator(name).id for name in gripper_actuator_names])
        self._gripper_opening_joint_id = self._model.joint('ur5e/robotiq/right_driver_joint').id

        # Modify PD gains.
        self._model.actuator_gainprm[self._arm_actuator_ids, 0] = np.asarray([4500, 4500, 4500, 2000, 2000, 500])
        self._model.actuator_gainprm[self._arm_actuator_ids, 2] = np.asarray([-450, -450, -450, -200, -200, -50])
        self._model.actuator_biasprm[self._arm_actuator_ids, 1] = -np.asarray([4500, 4500, 4500, 2000, 2000, 500])

        # Site IDs.
        self._pinch_site_id = self._model.site('ur5e/robotiq/pinch').id
        self._attach_site_id = self._model.site('ur5e/attachment_site').id

        pinch_pose = lie.SE3.from_rotation_and_translation(
            rotation=lie.SO3.from_matrix(self._data.site_xmat[self._pinch_site_id].reshape(3, 3)),
            translation=self._data.site_xpos[self._pinch_site_id],
        )
        attach_pose = lie.SE3.from_rotation_and_translation(
            rotation=lie.SO3.from_matrix(self._data.site_xmat[self._attach_site_id].reshape(3, 3)),
            translation=self._data.site_xpos[self._attach_site_id],
        )
        self._T_pa = pinch_pose.inverse() @ attach_pose

        self.post_compilation_objects()

    def post_compilation_objects(self):
        # Cube geom IDs.
        self._object_geom_ids_list = [
            [self._model.geom(geom.full_identifier).id for geom in object_geoms] for object_geoms in self._object_geoms_list
        ]
        self._object_target_mocap_ids = [
            self._model.body(f'object_target_{i}').mocapid[0] for i in range(self.num_objects)
        ]
        self._object_target_geom_ids_list = [
            [self._model.geom(geom.full_identifier).id for geom in object_target_geoms]
            for object_target_geoms in self._object_target_geoms_list
        ]

    def reset(self, options=None, *args, **kwargs):
        if 'task' in self._mode:
            if options is None:
                options = {}

            # Whether to provide a rendering of the goal.
            self._render_goal = False
            if 'render_goal' in options:
                self._render_goal = options['render_goal']

        return super().reset(*args, **kwargs)

    def initialize_arm(self):
        # Sample initial effector position and orientation.
        eff_pos = self.np_random.uniform(*self._arm_sampling_bounds)
        cur_ori = self._effector_down_rotation
        yaw = self.np_random.uniform(-np.pi, np.pi)
        rotz = lie.SO3.from_z_radians(yaw)
        eff_ori = rotz @ cur_ori

        # Solve for initial joint positions using IK.
        T_wp = lie.SE3.from_rotation_and_translation(eff_ori, eff_pos)
        T_wa = T_wp @ self._T_pa
        qpos_init = self._ik.solve(
            pos=T_wa.translation(),
            quat=T_wa.rotation().wxyz,
            curr_qpos=self._home_qpos,
        )

        self._data.qpos[self._arm_joint_ids] = qpos_init
        mujoco.mj_forward(self._model, self._data)

    def initialize_episode(self):
        # Set object colors.
        if self._mode == 'task_sort':
            color_idx = np.array([i % self._num_colors for i in range(self.num_objects)])
        elif self._random_colors:
            color_idx = np.random.choice(len(self._object_colors), size=self.num_objects, replace=False)
        else:
            color_idx = range(self.num_objects)
        for i in range(self.num_objects):
            for gid in self._object_geom_ids_list[i]:
                self._model.geom(gid).rgba = self._object_colors[color_idx[i]]
            for gid in self._object_target_geom_ids_list[i]:
                self._model.geom(gid).rgba[:3] = self._object_colors[color_idx[i], :3]

        self._data.qpos[self._arm_joint_ids] = self._home_qpos
        mujoco.mj_kinematics(self._model, self._data)

        if "data_collection" in self._mode:
            # Randomize the scene.
            self.initialize_arm()

            # Randomize object positions and orientations.
            state_xyzs = []
            for i in range(self.num_objects):
                obj_pos = self.sample_obj_pos(np.array(state_xyzs))
                state_xyzs.append(obj_pos)
                yaw = self.np_random.uniform(0, 2 * np.pi)
                obj_ori = lie.SO3.from_z_radians(yaw).wxyz.tolist()
                self._data.joint(f'object_joint_{i}').qpos[:3] = obj_pos
                self._data.joint(f'object_joint_{i}').qpos[3:] = obj_ori

            # Set a new target.
            self.set_new_target(return_info=False)
        else:
            saved_qpos = self._data.qpos.copy()
            saved_qvel = self._data.qvel.copy()
            
            # First, force set the current scene to a randomized goal state to obtain the goal observation.
            self.initialize_arm()
            goal_xyzs = []
            goal_oris = []
            for i in range(self.num_objects):
                obj_pos = self.sample_obj_pos(np.array(goal_xyzs), goal=True)
                goal_xyzs.append(obj_pos)
                yaw = self.np_random.uniform(0, 2 * np.pi)
                obj_ori = lie.SO3.from_z_radians(yaw).wxyz.tolist()
                goal_oris.append(obj_ori)
                self._data.joint(f'object_joint_{i}').qpos[:3] = obj_pos
                self._data.joint(f'object_joint_{i}').qpos[3:] = obj_ori
                self._data.mocap_pos[self._object_target_mocap_ids[i]] = obj_pos
                self._data.mocap_quat[self._object_target_mocap_ids[i]] = obj_ori
            mujoco.mj_forward(self._model, self._data)
            
            # Do a few random steps to make the scene stable.
            for _ in range(2):
                self.step(self.action_space.sample())

            # Save the goal observation.
            self._cur_goal_ob = self.compute_observation()
            self._cur_goal_ob_info = self.compute_ob_info()
            if self._render_goal:
                self._cur_goal_rendered = self.get_pixel_observation()
            else:
                self._cur_goal_rendered = None

            # Now, do the actual reset.
            self._data.qpos[:] = saved_qpos
            self._data.qvel[:] = saved_qvel
            self.initialize_arm()
            # Randomize object positions and orientations.
            state_xyzs = []
            for i in range(self.num_objects):
                obj_pos = self.sample_obj_pos(np.array(state_xyzs))
                state_xyzs.append(obj_pos)
                yaw = self.np_random.uniform(0, 2 * np.pi)
                obj_ori = lie.SO3.from_z_radians(yaw).wxyz.tolist()
                self._data.joint(f'object_joint_{i}').qpos[:3] = obj_pos
                self._data.joint(f'object_joint_{i}').qpos[3:] = obj_ori
                self._data.mocap_pos[self._object_target_mocap_ids[i]] = goal_xyzs[i]
                self._data.mocap_quat[self._object_target_mocap_ids[i]] = goal_oris[i]

        # Forward kinematics to update site positions.
        self.pre_step()
        mujoco.mj_forward(self._model, self._data)
        self.post_step()

        self._success = False
        self._success_frac = 0

    def sample_obj_pos(self, sampled_positions, goal=False):
        num_positions_sampled = sampled_positions.shape[0]
        
        if goal and self._mode == 'task_sort':
            if num_positions_sampled < self._num_colors:
                xy_pos = self._sort_goal_positions[num_positions_sampled]
                return (*xy_pos, 0.02)
            else:
                return (2, 2, 0.02)  # place the additional objects out of view
        
        else:
            if num_positions_sampled == 0:
                # First position is guaranteed to be collision-free
                xy_pos = self.np_random.uniform(*self._object_sampling_bounds)
                return (*xy_pos, 0.02)

            if goal and self._mode == 'task_stack' and num_positions_sampled % 2 == 1:
                xy_pos = sampled_positions[-1, :2]  # stack on top of the last odd object
                return (*xy_pos, 0.06)

            elif goal and self._mode == 'task_stack_all' and num_positions_sampled > 0:
                xy_pos = sampled_positions[0, :2]  # stack on top of the last object
                return (*xy_pos, 0.02 + 0.04 * num_positions_sampled)

        # Define minimum distance between objects
        cube_size = 0.04  # NOTE: maybe get from xml
        min_dist = 2 * cube_size * np.sqrt(2) / 2.0
                    
        # Sample collision-free positions
        valid = False
        for i in range(10000):
            # Sample xy positions
            xy_pos = self.np_random.uniform(*self._object_sampling_bounds)
            # Check if sampled values are valid
            xy_pos_copy = np.tile(xy_pos, (len(sampled_positions), 1))
            cube_dist = np.linalg.norm(xy_pos_copy - sampled_positions[..., :2], axis=-1)
            if np.all(cube_dist > min_dist, axis=-1):
                valid = True
                break
        
        # Make sure we succeeded at sampling
        assert valid, "Sampling cube locations was unsuccessful! ):"

        return (*xy_pos, 0.02)

    def set_new_target(self, return_info=True, p_stack=0.5):
        """Set a new random target for data collection.

        Args:
            return_info: Whether to return the observation and reset info.
            p_stack: Probability of stacking the target block on top of another block when there are multiple blocks.
        """
        assert 'data_collection' in self._mode

        block_xyzs = np.array([self._data.joint(f'object_joint_{i}').qpos[:3] for i in range(self.num_objects)])

        # Compute the top blocks.
        top_blocks = []
        for i in range(self.num_objects):
            for j in range(self.num_objects):
                if i == j:
                    continue
                if block_xyzs[j][2] > block_xyzs[i][2] and np.linalg.norm(block_xyzs[i][:2] - block_xyzs[j][:2]) < 0.02:
                    break
            else:
                top_blocks.append(i)

        if self._stacking:
            # Find the first block in the stack that is not in place
            next_target_index = None
            for idx, block_id in enumerate(self._stack_targets):
                block_pos = self._data.joint(f'object_joint_{block_id}').qpos[:3]
                tar_pos = self._stack_target_positions[idx]
                in_place = np.linalg.norm(block_pos - tar_pos) <= 0.02
                if not in_place:
                    next_target_index = idx
                    break
            assert next_target_index is not None
            self._stack_target_index = next_target_index
            self._target_block = self._stack_targets[self._stack_target_index]
            
        else:
            if self._mode == "data_collection_stack":
                if len(top_blocks) >= 2 and self.np_random.uniform() < p_stack:
                    # Multi-block stacking
                    self._stacking = True
                    stack_size = self.num_objects
                    # stack_size = self.np_random.integers(2, self.num_objects + 1)  # NOTE: uncomment to randomize stack size
                    self._stack_targets = self.np_random.choice(self.num_objects, size=stack_size, replace=False).tolist()
                    # Choose a random base position for the stack
                    base_pos = self.sample_obj_pos(np.array(block_xyzs))
                    self._stack_target_positions = [np.array([base_pos[0], base_pos[1], 0.02 + 0.04 * i]) for i in range(stack_size)]
                    self._stack_target_index = 0
                    self._stack_step_counter = 0
                    self._target_task = 'pickplace_cube'
                    self._target_block = self._stack_targets[self._stack_target_index]
                    tar_pos = self._stack_target_positions[self._stack_target_index]
                    yaw = self.np_random.uniform(0, 2 * np.pi)
                    tar_ori = lie.SO3.from_z_radians(yaw).wxyz.tolist()
                    self._stack_target_oris = [lie.SO3.from_z_radians(self.np_random.uniform(0, 2 * np.pi)).wxyz.tolist() for _ in range(stack_size)]
                    tar_ori = self._stack_target_oris[self._stack_target_index]
                else:
                    # Single block pickplace
                    self._stacking = False
                    self._target_block = self.np_random.choice(top_blocks)
                    self._target_task = 'pickplace_cube'
                    tar_pos = self.sample_obj_pos(np.array(block_xyzs))
                    yaw = self.np_random.uniform(0, 2 * np.pi)
                    tar_ori = lie.SO3.from_z_radians(yaw).wxyz.tolist()
            else:
                # Original logic for other data_collection modes
                self._target_block = self.np_random.choice(top_blocks)
                if self._target_block < self.num_cubes:
                    self._target_task = self.np_random.choice(['pickplace_cube', 'push_cube'], p=[0.3, 0.7])
                else:
                    self._target_task = self.np_random.choice(['pickplace_t', 'push_t'], p=[0.7, 0.3])
                self.stack = len(top_blocks) >= 2 and self.np_random.uniform() < p_stack
                if self.stack:
                    self.stack_block_idx = self.np_random.choice(list(set(top_blocks) - {self._target_block}))
                    block_pos = self._data.joint(f'object_joint_{self.stack_block_idx}').qpos[:3]
                    tar_pos = np.array([block_pos[0], block_pos[1], block_pos[2] + 0.04])
                else:
                    tar_pos = self.sample_obj_pos(np.array(block_xyzs))
                yaw = self.np_random.uniform(0, 2 * np.pi)
                tar_ori = lie.SO3.from_z_radians(yaw).wxyz.tolist()
        
        # Set mocap targets for all blocks
        for i in range(self.num_objects):
            if self._stacking and i in self._stack_targets:
                idx = self._stack_targets.index(i)
                self._data.mocap_pos[self._object_target_mocap_ids[i]] = self._stack_target_positions[idx]
                self._data.mocap_quat[self._object_target_mocap_ids[i]] = self._stack_target_oris[idx]
            elif not self._stacking and i == self._target_block:
                self._data.mocap_pos[self._object_target_mocap_ids[i]] = tar_pos
                self._data.mocap_quat[self._object_target_mocap_ids[i]] = tar_ori
            else:
                self._data.mocap_pos[self._object_target_mocap_ids[i]] = (0, 0, -0.3)
                self._data.mocap_quat[self._object_target_mocap_ids[i]] = lie.SO3.identity().wxyz.tolist()

        # Set the target colors.
        for i in range(self.num_objects):
            if self._visualize_info and ((self._stacking) or (not self._stacking and i == self._target_block)):
                for gid in self._object_target_geom_ids_list[i]:
                    self._model.geom(gid).rgba[3] = 0.2
            else:
                for gid in self._object_target_geom_ids_list[i]:
                    self._model.geom(gid).rgba[3] = 0.0

        if return_info:
            return self.compute_observation(), self.get_reset_info()

    def update_stack_target(self):
        # Only applies if stacking is active
        if not getattr(self, "_stacking", False):
            return

        # Find the first block in the stack that is not in place
        next_target_index = None
        for idx, block_id in enumerate(self._stack_targets):
            block_pos = self._data.joint(f'object_joint_{block_id}').qpos[:3]
            tar_pos = self._stack_target_positions[idx]
            in_place = np.linalg.norm(block_pos - tar_pos) <= 0.02
            if not in_place:
                next_target_index = idx
                break
        
        # If a previous block is not in place, make it your target. Otherwise, let set_new_target() handle the next block logic
        if next_target_index is not None:
            next_target_index = min(self._stack_target_index, next_target_index)

        self._stack_step_counter += 1

        # If all blocks are in place, or timeout, reset stack targets
        if next_target_index is None or self._stack_step_counter > 500:
            self._stacking = False
            return

        # Set the new target block
        self._stack_target_index = next_target_index
        self._target_block = self._stack_targets[self._stack_target_index]

    def set_control(self, action):
        action = self.unnormalize_action(action)
        a_pos, a_ori, a_gripper = action[:3], action[3], action[4]

        # Compute target effector pose based on the relative action.
        effector_pos = self._data.site_xpos[self._pinch_site_id].copy()
        effector_yaw = lie.SO3.from_matrix(
            self._data.site_xmat[self._pinch_site_id].copy().reshape(3, 3)
        ).compute_yaw_radians()
        gripper_opening = np.array(np.clip([self._data.qpos[self._gripper_opening_joint_id] / 0.8], 0, 1))
        target_effector_translation = effector_pos + a_pos
        target_effector_orientation = (
            lie.SO3.from_z_radians(a_ori)
            @ lie.SO3.from_z_radians(effector_yaw)
            @ self._effector_down_rotation.inverse()
        )
        target_gripper_opening = gripper_opening + a_gripper

        # Make sure the target pose respects the action limits.
        np.clip(
            target_effector_translation,
            *self._workspace_bounds,
            out=target_effector_translation,
        )
        yaw = np.clip(
            target_effector_orientation.compute_yaw_radians(),
            -np.pi,
            +np.pi,
        )
        target_effector_orientation = lie.SO3.from_z_radians(yaw) @ self._effector_down_rotation
        target_gripper_opening = np.clip(target_gripper_opening, 0.0, 1.0)

        # Pinch pose in the world frame -> attach pose in the world frame.
        self._target_effector_pose = lie.SE3.from_rotation_and_translation(
            rotation=target_effector_orientation,
            translation=target_effector_translation,
        )
        T_wa = self._target_effector_pose @ self._T_pa

        # Solve for the desired joint positions.
        qpos_target = self._ik.solve(
            pos=T_wa.translation(),
            quat=T_wa.rotation().wxyz,
            curr_qpos=self._data.qpos[self._arm_joint_ids],
        )

        # Set the desired joint positions for the underlying PD controller.
        self._data.ctrl[self._arm_actuator_ids] = qpos_target
        self._data.ctrl[self._gripper_actuator_ids] = 255.0 * target_gripper_opening

    def pre_step(self):
        self._prev_qpos = self._data.qpos.copy()
        self._prev_qvel = self._data.qvel.copy()
        self._prev_ob_info = self.compute_ob_info()

    def post_step(self):
        # Check if the objects are in the target configurations.
        object_successes = self._compute_successes()
        if 'data_collection' in self._mode:
            self.update_stack_target()
            self._success = object_successes[self._target_block]
        else:
            self._success = all(object_successes)
            self._success_frac = np.mean(object_successes)

        # Adjust the colors of the target cubes.
        for i in range(self.num_objects):
            if self._visualize_info and ('task' in self._mode or self._stacking or i == self._target_block):
                for gid in self._object_target_geom_ids_list[i]:
                    self._model.geom(gid).rgba[3] = 0.2
            else:
                for gid in self._object_target_geom_ids_list[i]:
                    self._model.geom(gid).rgba[3] = 0.0

    def compute_ob_info(self):
        ob_info = {}

        # Proprioceptive observations
        ob_info['proprio/joint_pos'] = self._data.qpos[self._arm_joint_ids].copy()
        ob_info['proprio/joint_vel'] = self._data.qvel[self._arm_joint_ids].copy()
        ob_info['proprio/effector_pos'] = self._data.site_xpos[self._pinch_site_id].copy()
        ob_info['proprio/effector_yaw'] = np.array(
            [lie.SO3.from_matrix(self._data.site_xmat[self._pinch_site_id].copy().reshape(3, 3)).compute_yaw_radians()]
        )
        ob_info['proprio/gripper_opening'] = np.array(
            np.clip([self._data.qpos[self._gripper_opening_joint_id] / 0.8], 0, 1)
        )
        ob_info['proprio/gripper_vel'] = self._data.qvel[[self._gripper_opening_joint_id]].copy()
        ob_info['proprio/gripper_contact'] = np.array(
            [np.clip(np.linalg.norm(self._data.body('ur5e/robotiq/right_pad').cfrc_ext) / 50, 0, 1)]
        )

        self.add_object_info(ob_info)

        ob_info['prev_qpos'] = self._prev_qpos.copy()
        ob_info['prev_qvel'] = self._prev_qvel.copy()
        ob_info['qpos'] = self._data.qpos.copy()
        ob_info['qvel'] = self._data.qvel.copy()
        ob_info['control'] = self._data.ctrl.copy()
        ob_info['time'] = np.array([self._data.time])

        return ob_info

    def add_object_info(self, ob_info):
        # Object positions and orientations.
        for i in range(self.num_objects):
            ob_info[f'privileged/block_{i}_pos'] = self._data.joint(f'object_joint_{i}').qpos[:3].copy()
            ob_info[f'privileged/block_{i}_quat'] = self._data.joint(f'object_joint_{i}').qpos[3:].copy()
            ob_info[f'privileged/block_{i}_yaw'] = np.array([lie.SO3(wxyz=self._data.joint(f'object_joint_{i}').qpos[3:]).compute_yaw_radians()])

        if 'data_collection' in self._mode:
            # Target cube info.
            ob_info['privileged/target_task'] = self._target_task

            target_mocap_id = self._object_target_mocap_ids[self._target_block]
            ob_info['privileged/target_block'] = self._target_block
            ob_info['privileged/target_block_pos'] = self._data.mocap_pos[target_mocap_id].copy()
            ob_info['privileged/target_block_yaw'] = np.array([lie.SO3(wxyz=self._data.mocap_quat[target_mocap_id]).compute_yaw_radians()])

    def get_pixel_observation(self):
        if self._mode in ['task_stack', 'task_stack_all', 'data_collection_stack']:
            camera = 'front_pixels' if not self._multiview else ['front_pixels', 'side_pixels_stack']
        else:
            camera = 'front_pixels' if not self._multiview else ['front_pixels', 'side_pixels']
        frame = self.render(camera=camera)
        return frame

    def compute_observation(self):
        if self._ob_type == 'pixels':
            return self.get_pixel_observation()
        else:
            ob_info = self.compute_ob_info()
            if self._ob_type == 'states_clean':
                return self.get_state_clean_observation(ob_info)
            else: # self._ob_type == 'states'
                return self.get_state_observation(ob_info)

    def get_state_clean_observation(self, ob_info):
        xyz_center = np.array([0.425, 0.0, 0.0])
        xyz_scaler = 10.0
        gripper_scaler = 3.0

        ob = [
                (ob_info['proprio/effector_pos'] - xyz_center) * xyz_scaler,
                np.cos(ob_info['proprio/effector_yaw']),
                np.sin(ob_info['proprio/effector_yaw']),
                ob_info['proprio/gripper_opening'] * gripper_scaler,
            ]
        for i in range(self.num_objects):
            ob.extend(
                [
                    (ob_info[f'privileged/block_{i}_pos'] - xyz_center) * xyz_scaler,
                    np.cos(ob_info[f'privileged/block_{i}_yaw']),
                    np.sin(ob_info[f'privileged/block_{i}_yaw']),
                ]
            )

        return np.concatenate(ob)
    
    def get_state_observation(self, ob_info):
        xyz_center = np.array([0.425, 0.0, 0.0])
        xyz_scaler = 10.0
        gripper_scaler = 3.0

        ob = [
                ob_info['proprio/joint_pos'],
                ob_info['proprio/joint_vel'],
                (ob_info['proprio/effector_pos'] - xyz_center) * xyz_scaler,
                np.cos(ob_info['proprio/effector_yaw']),
                np.sin(ob_info['proprio/effector_yaw']),
                ob_info['proprio/gripper_opening'] * gripper_scaler,
                ob_info['proprio/gripper_contact'],
            ]
        for i in range(self.num_objects):
            ob.extend(
                [
                    (ob_info[f'privileged/block_{i}_pos'] - xyz_center) * xyz_scaler,
                    ob_info[f'privileged/block_{i}_quat'],
                    np.cos(ob_info[f'privileged/block_{i}_yaw']),
                    np.sin(ob_info[f'privileged/block_{i}_yaw']),
                ]
            )

        return np.concatenate(ob)
    
    def _compute_successes(self):
        """Compute object successes."""
        successes = []

        if self._mode == 'task_sort':
            for i in range(self.num_cubes):
                obj_pos = self._data.joint(f'object_joint_{i}').qpos[:3]
                tar_pos = self._data.mocap_pos[self._object_target_mocap_ids[i % self._num_colors]]
                successes.append(np.linalg.norm(obj_pos - tar_pos) <= 4 * 0.04)

        else:
            for i in range(self.num_cubes):
                obj_pos = self._data.joint(f'object_joint_{i}').qpos[:3]
                tar_pos = self._data.mocap_pos[self._object_target_mocap_ids[i]]
                
                # Cube: only position matters
                pos_success = np.linalg.norm(obj_pos - tar_pos) <= 0.04
                successes.append(pos_success)

        return successes
    
    def compute_reward(self):
        
        if self._reward_mode == 'sparse':
            # Success if all cubes are in place
            successes = self._compute_successes()
            reward = 0.0 if all(successes) else -1.0

        elif self._reward_mode == 'step':
            # Penalize each object not in place
            successes = self._compute_successes()
            reward = -float(np.mean([not s for s in successes]))

        elif self._reward_mode == 'dense':
            # Penalize mean position error
            obj_pos = np.zeros((self.num_cubes, 3), dtype=np.float32)
            tar_pos = np.zeros((self.num_cubes, 3), dtype=np.float32)
            for i in range(self.num_cubes):
                obj_pos[i] = self._data.joint(f'object_joint_{i}').qpos[:3]
                tar_pos[i] = self._data.mocap_pos[self._object_target_mocap_ids[i]]
            dists = np.linalg.norm(obj_pos - tar_pos, axis=-1)
            reward = -np.mean(dists) * self._reward_scale
        
        else:
            raise NotImplementedError(f"Reward mode '{self._reward_mode}' is not implemented.")

        return reward

    def compute_reward_batch(self, qpos, goal_qpos):
        """Compute the reward for a batch of observations and goals from buffer."""
        qpos_obj_start_idx = 14
        qpos_obj_length = 7  # 3 pos + 4 quat

        obj_xyzs_list = []
        goal_xyzs_list = []
        obj_quats_list = []
        goal_quats_list = []
        
        for i in range(self.num_cubes):
            obj_xyzs_list.append(qpos[:, qpos_obj_start_idx + i * qpos_obj_length : qpos_obj_start_idx + i * qpos_obj_length + 3])
            goal_xyzs_list.append(goal_qpos[:, qpos_obj_start_idx + i * qpos_obj_length : qpos_obj_start_idx + i * qpos_obj_length + 3])
            obj_quats_list.append(qpos[:, qpos_obj_start_idx + i * qpos_obj_length + 3 : qpos_obj_start_idx + (i+1) * qpos_obj_length])
            goal_quats_list.append(goal_qpos[:, qpos_obj_start_idx + i * qpos_obj_length + 3 : qpos_obj_start_idx + (i+1) * qpos_obj_length])
        obj_xyzs = torch.stack(obj_xyzs_list, dim=1)
        goal_xyzs = torch.stack(goal_xyzs_list, dim=1)
        obj_quats = torch.stack(obj_quats_list, dim=1)
        goal_quats = torch.stack(goal_quats_list, dim=1)

        dists = torch.linalg.norm(goal_xyzs - obj_xyzs, dim=-1)  # (batch, horizon, num_objects)

        # Success: all cubes are in place
        all_success = dists <= 0.04
        
        if self._reward_mode == 'sparse':
            reward = torch.where(torch.all(all_success, dim=-1, keepdim=True), 0.0, -1.0)
        elif self._reward_mode == 'step':
            reward = -1.0 * torch.mean((~all_success).float(), dim=-1, keepdim=True)
        elif self._reward_mode == 'dense':
            reward = -torch.mean(dists, dim=-1, keepdim=True) * self.cfg.env_kwargs["reward_scale"]
        else:
            raise NotImplementedError(f"Reward mode '{self._reward_mode}' is not implemented.")

        return reward
    
    def get_reset_info(self):
        reset_info = self.compute_ob_info()
        reset_info['state_obs'] = self.get_state_clean_observation(reset_info)
        if 'task' in self._mode:
            reset_info['goal'] = self._cur_goal_ob
            reset_info['state_goal'] = self.get_state_clean_observation(self._cur_goal_ob_info)
            if self._render_goal is not None:
                reset_info['goal_rendered'] = self._cur_goal_rendered
        if 'data_collection' in self._mode:
            reset_info['prev_state_clean_observation'] = self.get_state_clean_observation(self._prev_ob_info)
        return reset_info

    def get_step_info(self):
        ob_info = self.compute_ob_info()
        ob_info['state_obs'] = self.get_state_clean_observation(ob_info)
        ob_info['success'] = self._success
        ob_info['success_frac'] = self._success_frac
        ob_info['was_agent_object_interaction'] = self._was_agent_object_interaction(ob_info)
        if 'data_collection' in self._mode:
            ob_info['prev_state_clean_observation'] = self.get_state_clean_observation(self._prev_ob_info)
        return ob_info

    def terminate_episode(self):
        if self._terminate_at_goal:
            return self._success
        else:
            return False

    def render(
        self,
        camera='front_pixels',
        *args,
        **kwargs,
    ):
        if isinstance(camera, (list, tuple)):
            imgs = []
            for cam in camera:
                img = super().render(camera=cam, *args, **kwargs)
                imgs.append(img)
            return np.stack(imgs, axis=0)
        else:
            return super().render(camera=camera, *args, **kwargs)

    def _was_agent_object_interaction(self, ob_info):
        qpos_obj_start_idx = 14
        qpos_object_length = 7

        prev_qpos = ob_info['prev_qpos']
        qpos = ob_info['qpos']

        prev_object_xyzs_list = []
        object_xyzs_list = []
        for i in range(self.num_objects):
            prev_object_xyzs_list.append(prev_qpos[..., qpos_obj_start_idx + i * qpos_object_length : qpos_obj_start_idx + i * qpos_object_length + 3])
            object_xyzs_list.append(qpos[..., qpos_obj_start_idx + i * qpos_object_length : qpos_obj_start_idx + i * qpos_object_length + 3])
        prev_object_xyzs = np.stack(prev_object_xyzs_list, axis=0)
        object_xyzs = np.stack(object_xyzs_list, axis=0)

        dists = np.linalg.norm(prev_object_xyzs - object_xyzs, axis=-1)
        return np.any(dists >= 1e-4, axis=-1)
    