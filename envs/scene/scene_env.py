from pathlib import Path

import gymnasium as gym
from gymnasium.spaces import Box
import mujoco

import numpy as np
import torch

from dm_control import mjcf

import ogbench
from ogbench.manipspace import controllers, lie, mjcf_utils
from ogbench.manipspace.envs.env import CustomMuJoCoEnv


class SceneEnv(CustomMuJoCoEnv):
    """Scene environment modified from OGBench.

    It contains a UR5e robot arm with a Robotiq 2F-85 gripper.
    The default control mode is relative end-effector control. The 5-D action space corresponds to the following:
    - 3-D relative end-effector position (x, y, z).
    - 1-D relative end-effector yaw.
    - 1-D relative gripper opening.

    This environment consists of a cube, two buttons, a drawer, and a window. The goal is to manipulate the objects
    to a target configuration. The buttons toggle the lock state of the drawer and window.
    """

    def __init__(
        self,
        ob_type='states',
        multiview=False,
        mode='task',
        reward_mode='sparse',
        num_cubes=1,
        terminate_at_goal=False,
        visualize_info=False,
        permute_blocks=True,
        reward_task_id=None,
        use_oracle_rep=False,
        physics_timestep=0.002,
        control_timestep=0.05,
        **kwargs,
    ):
        """Initialize the ManipSpace environment.

        Args:
            ob_type: Observation type. Either 'states' or 'pixels'.
            multiview: Whether to use multiview pixel observations.
            mode: Mode of the environment. Either 'task' or 'data_collection'.
                In 'task' mode, the environment is used for training and evaluation.
                In 'data_collection' mode, the environment is used for collecting offline data.
            reward_mode: Either 'sparse' or 'step'.
                'sparse': reward of 0 when the task is completed, -1 otherwise.
                'step': stepwise sparse reward for each completed subtask. Reward is -n/N, where n is the number of remaining subtasks and N is the total number of subtasks.
            num_cubes: Number of cubes in the environment (currently 1 or 2 is supported).
            terminate_at_goal: Whether to terminate the episode when the goal is reached.
            visualize_info: Whether to visualize the task information (e.g., success status).
            reward_task_id: Task ID for single-task RL. If this is not None, the environment operates in a single-task
            mode with the specified task ID. The task ID must be either a valid task ID or 0, where 0 means using the
            default task.
            use_oracle_rep: Whether to use oracle goal representations.
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
        self._colors = dict(
            red=np.array([0.96, 0.26, 0.33, 1.0]),
            orange=np.array([1.0, 0.69, 0.21, 1.0]),
            green=np.array([0.06, 0.74, 0.21, 1.0]),
            blue=np.array([0.35, 0.55, 0.91, 1.0]),
            purple=np.array([0.61, 0.28, 0.82, 1.0]),
            lightred=np.array([0.99, 0.85, 0.86, 1.0]),
            lightorange=np.array([1.0, 0.94, 0.84, 1.0]),
            lightgreen=np.array([0.77, 0.95, 0.81, 1.0]),
            lightblue=np.array([0.86, 0.9, 0.98, 1.0]),
            lightpurple=np.array([0.91, 0.84, 0.96, 1.0]),
            white=np.array([0.9, 0.9, 0.9, 1.0]),
            lightgray=np.array([0.7, 0.7, 0.7, 1.0]),
            gray=np.array([0.5, 0.5, 0.5, 1.0]),
            darkgray=np.array([0.3, 0.3, 0.3, 1.0]),
            black=np.array([0.1, 0.1, 0.1, 1.0]),
        )

        self._ob_type = ob_type
        self._multiview = multiview
        self._mode = mode
        self._reward_mode = reward_mode
        self._terminate_at_goal = terminate_at_goal
        self._visualize_info = visualize_info
        self._permute_blocks = permute_blocks
        self._reward_task_id = reward_task_id
        self._use_oracle_rep = use_oracle_rep

        assert ob_type in ['states', 'pixels']

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

        if self._mode == 'task':
            # Set task goals.
            self.task_infos = []
            self.cur_task_id = None
            self.cur_task_info = None
            self.set_tasks()
            self.num_tasks = len(self.task_infos)

            self._cur_goal_ob = None
            self._cur_goal_rendered = None
            self._render_goal = False

        self._success = False
        self._success_frac = 0.0

        # NOTE: from here on is scene-env specific init
        # Adjust workspace bounds to a smaller region.
        self._arm_sampling_bounds = np.asarray([[0.25, -0.2, 0.20], [0.40, 0.2, 0.35]])  # NOTE: modified arm sampling bounds to avoid goal occlusion
        self._object_sampling_bounds = np.asarray([[0.3, -0.07], [0.45, 0.18]])
        self._target_sampling_bounds = self._object_sampling_bounds

        # Define constants.
        self._drawer_center = np.array([0.33, -0.24, 0.066])
        self._cube_colors = np.array([self._colors['red'], self._colors['blue']])
        self._cube_success_colors = np.array([self._colors['lightred'], self._colors['lightblue']])
        self._num_cubes = num_cubes
        self._num_buttons = 2
        self._num_button_states = 2
        self._cur_button_states = np.array([0] * self._num_buttons)

        # Target info.
        self._target_task = 'cube'
        # The target cube position is stored in the mocap object.
        self._target_block = 0
        self._target_button = 0
        self._target_button_states = np.array([0] * self._num_buttons)
        self._target_drawer_pos = 0.0
        self._target_window_pos = 0.0

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

    def set_state(self, qpos, qvel, button_states):
        self._cur_button_states = button_states.copy()
        self._apply_button_states()
        super().set_state(qpos, qvel)

    def set_tasks(self):
        self.task_infos = [
            dict(
                # DESCRIPTION
                # tasks: open-drawer | open-window
                # total atomic tasks: 2
                # longest sequential dependency: 1
                task_name='task1_open',
                init=dict(
                    block_xyzs=np.array([[0.35, 0.05, 0.02], [0.35, -0.05, 0.02]]),
                    button_states=np.array([1, 1]),
                    drawer_pos=0.0,
                    window_pos=0.0,
                ),
                goal=dict(
                    block_xyzs=np.array([[0.35, 0.05, 0.02], [0.35, -0.05, 0.02]]),
                    button_states=np.array([1, 1]),
                    drawer_pos=-0.16,
                    window_pos=0.2,
                ),
            ),
            dict(
                # DESCRIPTION
                # tasks: unlock-drawer -> close-drawer -> lock-drawer | unlock-window -> close-window -> lock-window
                # total atomic tasks: 6
                # longest sequential dependency: 3
                task_name='task2_unlock_and_lock',
                init=dict(
                    block_xyzs=np.array([[0.35, -0.05, 0.02], [0.35, 0.05, 0.02]]),
                    button_states=np.array([0, 0]),
                    drawer_pos=-0.16,
                    window_pos=0.2,
                ),
                goal=dict(
                    block_xyzs=np.array([[0.35, -0.05, 0.02], [0.35, 0.05, 0.02]]),
                    button_states=np.array([0, 0]),
                    drawer_pos=0.0,
                    window_pos=0.0,
                ),
            ),
            dict(
                # DESCRIPTION
                # tasks: open-drawer | unlock-window -> close-window | move-cube-to-side
                # total atomic tasks: 4
                # longest sequential dependency: 2
                task_name='task3_rearrange_medium',
                init=dict(
                    block_xyzs=np.array([[0.4, 0.0, 0.02], [0.3, 0.0, 0.02]]),
                    button_states=np.array([1, 0]),
                    drawer_pos=0.0,
                    window_pos=0.2,
                ),
                goal=dict(
                    block_xyzs=np.array([[0.4, 0.15, 0.02], [0.3, 0.15, 0.02]]),
                    button_states=np.array([1, 1]),
                    drawer_pos=-0.16,
                    window_pos=0.0,
                ),
            ),
            dict(
                # DESCRIPTION
                # tasks: open-drawer -> lock-drawer & move-cube-to-drawer
                # total atomic tasks: 3
                # longest sequential dependency: 2
                task_name='task4_put_in_drawer',
                init=dict(
                    block_xyzs=np.array([[0.35, 0.05, 0.02], [0.35, 0.15, 0.02]]),
                    button_states=np.array([0, 0]),
                    drawer_pos=0.0,
                    window_pos=0.0,
                ),
                goal=dict(
                    block_xyzs=np.array([[0.33, -0.256, 0.065986], [0.35, 0.05, 0.02]]),
                    button_states=np.array([0, 0]),
                    drawer_pos=-0.16,
                    window_pos=0.0,
                ),
            ),
            dict(
                # DESCRIPTION
                # tasks: unlock-drawer -> open-drawer -> lock-drawer & move-cube-to-drawer | unlock-window -> open-window -> lock-window
                # total atomic tasks: 7
                # longest sequential dependency: 3
                task_name='task5_rearrange_hard',
                init=dict(
                    block_xyzs=np.array([[0.35, 0.15, 0.02], [0.35, 0.05, 0.02]]),
                    button_states=np.array([0, 0]),
                    drawer_pos=0.0,
                    window_pos=0.0,
                ),
                goal=dict(
                    block_xyzs=np.array([[0.33, -0.256, 0.065986], [0.35, 0.15, 0.02]]),
                    button_states=np.array([0, 0]),
                    drawer_pos=-0.16,
                    window_pos=0.2,
                ),
            ),
        ]

        if self._reward_task_id == 0:
            self._reward_task_id = 2  # Default task.

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
        # Add objects to scene.
        button_mjcf = mjcf.from_path((self._desc_dir / 'buttons.xml').as_posix())
        arena_mjcf.include_copy(button_mjcf)
        drawer_mjcf = mjcf.from_path((self._desc_dir / 'drawer.xml').as_posix())
        arena_mjcf.include_copy(drawer_mjcf)
        window_mjcf = mjcf.from_path((self._desc_dir / 'window.xml').as_posix())
        arena_mjcf.include_copy(window_mjcf)

        # Add cube scene.
        cube_outer_mjcf = mjcf.from_path((self._desc_dir / 'cube_outer.xml').as_posix())
        arena_mjcf.include_copy(cube_outer_mjcf)

        # Add `num_cubes` cubes to the scene.
        distance = 0.05
        for i in range(self._num_cubes):
            cube_mjcf = mjcf.from_path((self._desc_dir / 'cube_inner.xml').as_posix())
            pos = -distance * (self._num_cubes - 1) + 2 * distance * i
            cube_mjcf.find('body', 'object_0').pos[1] = pos
            cube_mjcf.find('body', 'object_target_0').pos[1] = pos
            for tag in ['body', 'joint', 'geom', 'site']:
                for item in cube_mjcf.find_all(tag):
                    if hasattr(item, 'name') and item.name is not None and item.name.endswith('_0'):
                        item.name = item.name[:-2] + f'_{i}'
            arena_mjcf.include_copy(cube_mjcf)

        # Save geoms.
        self._cube_geoms_list = []
        for i in range(self._num_cubes):
            self._cube_geoms_list.append(arena_mjcf.find('body', f'object_{i}').find_all('geom'))
        self._cube_target_geoms_list = []
        for i in range(self._num_cubes):
            self._cube_target_geoms_list.append(arena_mjcf.find('body', f'object_target_{i}').find_all('geom'))
        self._button_geoms_list = []
        for i in range(self._num_buttons):
            self._button_geoms_list.append([button_mjcf.find('geom', f'btngeom_{i}')])

        # Add cameras.
        cameras = {
            'front_pixels': {
                'pos': (0.905, 0.000, 0.762),
                'xyaxes': (0.000, 1.000, 0.000, -0.771, 0.000, 0.637),
            },
            'side_pixels': {
                'pos': (0.400, -0.505, 0.762),
                'xyaxes': (1.000, 0.000, 0.000, 0.00, 0.771, 0.637),
            },
        }
        for camera_name, camera_kwargs in cameras.items():
            arena_mjcf.worldbody.add('camera', name=camera_name, **camera_kwargs)

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
        self._cube_geom_ids_list = [
            [self._model.geom(geom.full_identifier).id for geom in cube_geoms] for cube_geoms in self._cube_geoms_list
        ]
        self._cube_target_mocap_ids = [
            self._model.body(f'object_target_{i}').mocapid[0] for i in range(self._num_cubes)
        ]
        self._cube_target_geom_ids_list = [
            [self._model.geom(geom.full_identifier).id for geom in cube_target_geoms]
            for cube_target_geoms in self._cube_target_geoms_list
        ]

        # Button geom IDs.
        self._button_geom_ids_list = [
            [self._model.geom(geom.full_identifier).id for geom in button_geoms]
            for button_geoms in self._button_geoms_list
        ]
        self._button_site_ids = [self._model.site(f'btntop_{i}').id for i in range(self._num_buttons)]

        # Drawer and window site IDs.
        self._drawer_site_id = self._model.site('drawer_handle_center').id
        self._drawer_target_site_id = self._model.site('drawer_handle_center_target').id

        self._window_site_id = self._model.site('window_handle_center').id
        self._window_target_site_id = self._model.site('window_handle_center_target').id

    def _apply_button_states(self):
        # Adjust button colors based on the current state.
        for i in range(self._num_buttons):
            for gid in self._button_geom_ids_list[i]:
                self._model.geom(gid).rgba = self._colors['red' if self._cur_button_states[i] == 0 else 'white']

        # Lock or unlock the drawer and window based on the button states.
        # We adjust the damping of the joints to lock the drawer and window. This needs to be set carefully because
        # setting it to a very high value can cause numerical instability. We use 1e6. This is a reasonably safe value,
        # but it still allows the drawer and window to move very slightly with a strong enough force. We also tested
        # 1e7, but it caused numerical instability when interacting with the cube.
        if self._cur_button_states[0] == 0:
            # Set the damping to a high value to lock the drawer.
            self._model.joint('drawer_slide').damping[0] = 1e6
            self._model.material('drawer_handle').rgba = self._colors['red']
        else:
            # Unset the damping to unlock the drawer.
            self._model.joint('drawer_slide').damping[0] = 2.0
            self._model.material('drawer_handle').rgba = self._colors['white']
        if self._cur_button_states[1] == 0:
            # Set the damping to a high value to lock the window.
            self._model.joint('window_slide').damping[0] = 1e6
            self._model.material('window_handle').rgba = self._colors['red']
        else:
            # Unset the damping to unlock the window.
            self._model.joint('window_slide').damping[0] = 2.0
            self._model.material('window_handle').rgba = self._colors['white']

        mujoco.mj_forward(self._model, self._data)
    
    def reset(self, options=None, *args, **kwargs):
        if self._mode == 'task':
            # Set the task goal.
            if options is None:
                options = {}

            if self._reward_task_id is not None:
                # Use the pre-defined task.
                assert 1 <= self._reward_task_id <= self.num_tasks, f'Task ID must be in [1, {self.num_tasks}].'
                self.cur_task_id = self._reward_task_id
                self.cur_task_info = self.task_infos[self.cur_task_id - 1]
            elif 'task_id' in options:
                # Use the pre-defined task.
                assert 1 <= options['task_id'] <= self.num_tasks, f'Task ID must be in [1, {self.num_tasks}].'
                self.cur_task_id = options['task_id']
                self.cur_task_info = self.task_infos[self.cur_task_id - 1]
            elif 'task_info' in options:
                # Use the provided task information.
                self.cur_task_id = None
                self.cur_task_info = options['task_info']
            else:
                # Randomly sample a task.
                self.cur_task_id = np.random.randint(1, self.num_tasks + 1)
                self.cur_task_info = self.task_infos[self.cur_task_id - 1]

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
        # Set cube colors.
        for i in range(self._num_cubes):
            for gid in self._cube_geom_ids_list[i]:
                self._model.geom(gid).rgba = self._cube_colors[i]
            for gid in self._cube_target_geom_ids_list[i]:
                self._model.geom(gid).rgba[:3] = self._cube_colors[i, :3]

        self._data.qpos[self._arm_joint_ids] = self._home_qpos
        mujoco.mj_kinematics(self._model, self._data)

        if self._mode == 'data_collection':
            # Randomize the scene.

            self.initialize_arm()

            # Randomize block positions and orientations.
            for i in range(self._num_cubes):
                xy = self.np_random.uniform(*self._object_sampling_bounds)
                obj_pos = (*xy, 0.02)
                yaw = self.np_random.uniform(0, 2 * np.pi)
                obj_ori = lie.SO3.from_z_radians(yaw).wxyz.tolist()
                self._data.joint(f'object_joint_{i}').qpos[:3] = obj_pos
                self._data.joint(f'object_joint_{i}').qpos[3:] = obj_ori

            # Randomize button states.
            for i in range(self._num_buttons):
                self._cur_button_states[i] = self.np_random.choice(self._num_button_states)
            self._apply_button_states()

            # Randomize drawer and window positions.
            self._data.joint('drawer_slide').qpos[0] = self.np_random.uniform(-0.16, 0)
            self._data.joint('window_slide').qpos[0] = self.np_random.uniform(0, 0.2)

            # Set a new target.
            self.set_new_target(return_info=False)
        else:
            # Set object positions and orientations based on the current task.

            if self._permute_blocks:
                # Randomize the order of the cubes when there are multiple cubes.
                block_permutation = self.np_random.permutation(self._num_cubes)
            else:
                block_permutation = np.arange(self._num_cubes)
            init_block_xyzs = self.cur_task_info['init']['block_xyzs'].copy()[block_permutation]
            goal_block_xyzs = self.cur_task_info['goal']['block_xyzs'].copy()[block_permutation]
            # Get the current task info for the other objects.
            init_button_states = self.cur_task_info['init']['button_states'].copy()
            goal_button_states = self.cur_task_info['goal']['button_states'].copy()
            init_drawer_pos = self.cur_task_info['init']['drawer_pos']
            goal_drawer_pos = self.cur_task_info['goal']['drawer_pos']
            init_window_pos = self.cur_task_info['init']['window_pos']
            goal_window_pos = self.cur_task_info['goal']['window_pos']

            # First, force set the current scene to the goal state to obtain the goal observation.
            saved_qpos = self._data.qpos.copy()
            saved_qvel = self._data.qvel.copy()
            self.initialize_arm()
            for i in range(self._num_cubes):
                self._data.joint(f'object_joint_{i}').qpos[:3] = goal_block_xyzs[i]
                self._data.joint(f'object_joint_{i}').qpos[3:] = lie.SO3.identity().wxyz.tolist()
                self._data.mocap_pos[self._cube_target_mocap_ids[i]] = goal_block_xyzs[i]
                self._data.mocap_quat[self._cube_target_mocap_ids[i]] = lie.SO3.identity().wxyz.tolist()
            self._cur_button_states = goal_button_states.copy()
            self._apply_button_states()
            self._data.joint('drawer_slide').qpos[0] = goal_drawer_pos
            self._data.joint('window_slide').qpos[0] = goal_window_pos
            mujoco.mj_forward(self._model, self._data)

            # Do a few random steps to make the scene stable.
            for _ in range(2):
                self.step(self.action_space.sample())

            # Save the goal observation.
            self._cur_goal_ob = (
                self.compute_oracle_observation() if self._use_oracle_rep else self.compute_observation()
            )
            if self._render_goal:
                self._cur_goal_rendered = self.get_pixel_observation()
            else:
                self._cur_goal_rendered = None

            # Now, do the actual reset.
            self._data.qpos[:] = saved_qpos
            self._data.qvel[:] = saved_qvel
            self.initialize_arm()
            for i in range(self._num_cubes):
                # Randomize the position and orientation of the cube slightly.
                obj_pos = init_block_xyzs[i].copy()
                obj_pos[:2] += self.np_random.uniform(-0.01, 0.01, size=2)
                self._data.joint(f'object_joint_{i}').qpos[:3] = obj_pos
                yaw = self.np_random.uniform(0, 2 * np.pi)
                obj_ori = lie.SO3.from_z_radians(yaw).wxyz.tolist()
                self._data.joint(f'object_joint_{i}').qpos[3:] = obj_ori
                self._data.mocap_pos[self._cube_target_mocap_ids[i]] = goal_block_xyzs[i]
                self._data.mocap_quat[self._cube_target_mocap_ids[i]] = lie.SO3.identity().wxyz.tolist()
            # Set the button states.
            self._cur_button_states = init_button_states.copy()
            self._target_button_states = goal_button_states.copy()
            self._apply_button_states()
            # Randomize the drawer and window positions slightly.
            self._data.joint('drawer_slide').qpos[0] = np.clip(
                init_drawer_pos + self.np_random.uniform(-0.01, 0.01), -0.16, 0
            )
            self._model.site('drawer_handle_center_target').pos[1] = goal_drawer_pos
            self._target_drawer_pos = goal_drawer_pos
            self._data.joint('window_slide').qpos[0] = np.clip(
                init_window_pos + self.np_random.uniform(-0.01, 0.01), 0, 0.2
            )
            self._model.site('window_handle_center_target').pos[0] = goal_window_pos
            self._target_window_pos = goal_window_pos

        # Forward kinematics to update site positions.
        self.pre_step()
        mujoco.mj_forward(self._model, self._data)
        self.post_step()

        self._success = False
        self._success_frac = 0.0

    def _is_in_drawer(self, obj_pos):
        """Check if the object is in the drawer."""
        drawer_pos_y = self._data.site_xpos[self._drawer_site_id][1]
        drawer_low = np.array([0.21, drawer_pos_y - 0.27, 0.0])
        drawer_high = np.array([0.45, drawer_pos_y - 0.07, 0.15])
        return np.all(drawer_low <= obj_pos) and np.all(obj_pos <= drawer_high)

    def set_new_target(self, return_info=True, p_stack=0.5):
        """Set a new random target for data collection.

        Args:
            return_info: Whether to return the observation and reset info.
            p_stack: Probability of stacking the target block on top of another block when there are multiple blocks
                and the target task is 'cube'.
        """
        assert self._mode == 'data_collection'

        # Only consider blocks not in the drawer.
        available_blocks = []
        for i in range(self._num_cubes):
            if not self._is_in_drawer(self._data.joint(f'object_joint_{i}').qpos[:3]):
                available_blocks.append(i)

        # Probability of each task.
        p_cube = 1.0 if len(available_blocks) > 0 else 0.0
        p_button = 1.0
        p_drawer = 0.25 if self._cur_button_states[0] == 0 else 1.0
        p_window = 0.25 if self._cur_button_states[1] == 0 else 1.0
        probs = np.array([p_cube, p_button, p_drawer, p_window])
        probs /= probs.sum()

        # Probability of putting the target block in the drawer when the target task is 'cube'.
        p_put_in_drawer = 0.3

        self._target_task = self.np_random.choice(['cube', 'button', 'drawer', 'window'], p=probs)

        if self._target_task == 'cube':
            # Set cube target.
            block_xyzs = np.array([self._data.joint(f'object_joint_{i}').qpos[:3] for i in range(self._num_cubes)])

            # Compute the top blocks.
            top_blocks = []
            for i in range(self._num_cubes):
                if i not in available_blocks:
                    continue
                for j in range(self._num_cubes):
                    if i == j:
                        continue
                    if (
                        block_xyzs[j][2] > block_xyzs[i][2]
                        and np.linalg.norm(block_xyzs[i][:2] - block_xyzs[j][:2]) < 0.02
                    ):
                        break
                else:
                    top_blocks.append(i)

            # Pick one of the top cubes as the target.
            self._target_block = self.np_random.choice(top_blocks)

            drawer_is_open = self._data.joint('drawer_slide').qpos[0] < -0.12
            drawer_is_empty = len(available_blocks) == 2  # NOTE: this is an additional condition for 2 cubes not in the original OGBench
            put_in_drawer = drawer_is_open and drawer_is_empty and self.np_random.uniform() < p_put_in_drawer
            
            stack = len(top_blocks) >= 2 and self.np_random.uniform() < p_stack
            if put_in_drawer:
                # Put the target block in the drawer.
                tar_pos = self._drawer_center.copy()
                tar_pos[:2] = tar_pos[:2] + self.np_random.uniform(-0.005, 0.005, size=2)
            elif stack:
                # Stack the target block on top of another block.
                block_idx = self.np_random.choice(list(set(top_blocks) - {self._target_block}))
                block_pos = self._data.joint(f'object_joint_{block_idx}').qpos[:3]
                tar_pos = np.array([block_pos[0], block_pos[1], block_pos[2] + 0.04])
            else:
                # Randomize target position.
                xy = self.np_random.uniform(*self._target_sampling_bounds)
                tar_pos = (*xy, 0.02)
            # Randomize target orientation.
            yaw = self.np_random.uniform(0, 2 * np.pi)
            tar_ori = lie.SO3.from_z_radians(yaw).wxyz.tolist()

            # Only show the target block.
            for i in range(self._num_cubes):
                if i == self._target_block:
                    # Set the target position and orientation.
                    self._data.mocap_pos[self._cube_target_mocap_ids[i]] = tar_pos
                    self._data.mocap_quat[self._cube_target_mocap_ids[i]] = tar_ori
                else:
                    # Move the non-target blocks out of the way.
                    self._data.mocap_pos[self._cube_target_mocap_ids[i]] = (0, 0, -0.3)
                    self._data.mocap_quat[self._cube_target_mocap_ids[i]] = lie.SO3.identity().wxyz.tolist()

            # Set the target colors.
            for i in range(self._num_cubes):
                if self._visualize_info and i == self._target_block:
                    for gid in self._cube_target_geom_ids_list[i]:
                        self._model.geom(gid).rgba[3] = 0.2
                else:
                    for gid in self._cube_target_geom_ids_list[i]:
                        self._model.geom(gid).rgba[3] = 0.0
        elif self._target_task == 'button':
            # Set target button.
            self._target_button = self.np_random.choice(self._num_buttons)
            self._target_button_states[self._target_button] = (
                self._cur_button_states[self._target_button] + 1
            ) % self._num_button_states
        elif self._target_task == 'drawer':
            # Set target drawer position.
            if self._data.joint('drawer_slide').qpos[0] >= -0.08:  # Drawer closed.
                self._target_drawer_pos = -0.16
            else:  # Drawer open.
                self._target_drawer_pos = 0.0
            self._model.site('drawer_handle_center_target').pos[1] = self._target_drawer_pos
        elif self._target_task == 'window':
            # Set target window position.
            if self._data.joint('window_slide').qpos[0] <= 0.1:  # Window closed.
                self._target_window_pos = 0.2
            else:  # Window open.
                self._target_window_pos = 0.0
            self._model.site('window_handle_center_target').pos[0] = self._target_window_pos

        mujoco.mj_kinematics(self._model, self._data)

        if return_info:
            return self.compute_observation(), self.get_reset_info()

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
        self._prev_button_states = self._cur_button_states.copy()
        self._prev_qpos = self._data.qpos.copy()
        self._prev_qvel = self._data.qvel.copy()
        self._prev_ob_info = self.compute_ob_info()

    def _compute_successes(self):
        """Compute object successes."""
        cube_successes = []
        for i in range(self._num_cubes):
            obj_pos = self._data.joint(f'object_joint_{i}').qpos[:3]
            tar_pos = self._data.mocap_pos[self._cube_target_mocap_ids[i]]
            if np.linalg.norm(obj_pos - tar_pos) <= 0.04:
                cube_successes.append(True)
            else:
                cube_successes.append(False)
        button_successes = [
            (self._cur_button_states[i] == self._target_button_states[i]) for i in range(self._num_buttons)
        ]
        drawer_success = np.abs(self._data.joint('drawer_slide').qpos[0] - self._target_drawer_pos) <= 0.04
        window_success = np.abs(self._data.joint('window_slide').qpos[0] - self._target_window_pos) <= 0.04

        return cube_successes, button_successes, drawer_success, window_success
    
    def post_step(self):
        # Check numerical stability.
        if self._mode == 'task':
            # Very rarely, the blocks can go out of bounds due to numerical instability. This can (rarely) happen
            # when the robot presses the drawer lock button while the drawer is moving and the block is in the drawer.
            # We only check this in the task mode, because we will manually filter out these cases outside the class in
            # the data collection mode with a more stringent check.
            is_healthy = True
            for i in range(self._num_cubes):
                obj_pos = self._data.joint(f'object_joint_{i}').qpos[:3]
                # Check if the object is out of bounds.
                if np.any(obj_pos <= self._workspace_bounds[0] - 0.2) or np.any(
                    obj_pos >= self._workspace_bounds[1] + 0.2
                ):
                    is_healthy = False
                    break

            if not is_healthy:
                # Manually reset the cube position to a random initial position.
                print('Numerical instability detected. Resetting cube positions.', flush=True)
                for i in range(self._num_cubes):
                    xy = self.np_random.uniform(*self._object_sampling_bounds)
                    obj_pos = (*xy, 0.02)
                    yaw = self.np_random.uniform(0, 2 * np.pi)
                    obj_ori = lie.SO3.from_z_radians(yaw).wxyz.tolist()
                    self._data.joint(f'object_joint_{i}').qpos[:3] = obj_pos
                    self._data.joint(f'object_joint_{i}').qpos[3:] = obj_ori
                    self._data.joint('object_joint_0').qvel[:] = 0.0
                mujoco.mj_forward(self._model, self._data)

        # Update button states.
        for i in range(self._num_buttons):
            prev_joint_pos = self._prev_ob_info[f'privileged/button_{i}_pos'][0]
            cur_joint_pos = self._data.joint(f'buttonbox_joint_{i}').qpos.copy()[0]
            if prev_joint_pos > -0.02 and cur_joint_pos <= -0.02:
                # Button pressed: change the state of the button.
                self._cur_button_states[i] = (self._cur_button_states[i] + 1) % self._num_button_states
        self._apply_button_states()

        # Evaluate successes.
        cube_successes, button_successes, drawer_success, window_success = self._compute_successes()
        if self._mode == 'data_collection':
            self._success = {
                'cube': cube_successes[self._target_block],
                'button': button_successes[self._target_button],
                'drawer': drawer_success,
                'window': window_success,
            }[self._target_task]
        else:
            self._success = all(cube_successes) and all(button_successes) and drawer_success and window_success
            self._success_frac = np.mean(cube_successes + button_successes + [drawer_success, window_success])

        # Adjust the colors of the cubes based on success.
        for i in range(self._num_cubes):
            if self._visualize_info and (self._mode == 'task' or i == self._target_block):
                for gid in self._cube_target_geom_ids_list[i]:
                    self._model.geom(gid).rgba[3] = 0.2
            else:
                for gid in self._cube_target_geom_ids_list[i]:
                    self._model.geom(gid).rgba[3] = 0.0

            if self._visualize_info and cube_successes[i]:
                for gid in self._cube_geom_ids_list[i]:
                    self._model.geom(gid).rgba[:3] = self._cube_success_colors[i, :3]
            else:
                for gid in self._cube_geom_ids_list[i]:
                    self._model.geom(gid).rgba[:3] = self._cube_colors[i, :3]
    
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
        # Cube positions and orientations.
        for i in range(self._num_cubes):
            ob_info[f'privileged/block_{i}_pos'] = self._data.joint(f'object_joint_{i}').qpos[:3].copy()
            ob_info[f'privileged/block_{i}_quat'] = self._data.joint(f'object_joint_{i}').qpos[3:].copy()
            ob_info[f'privileged/block_{i}_yaw'] = np.array(
                [lie.SO3(wxyz=self._data.joint(f'object_joint_{i}').qpos[3:]).compute_yaw_radians()]
            )

        # Button states.
        for i in range(self._num_buttons):
            ob_info[f'privileged/button_{i}_state'] = self._cur_button_states[i]
            ob_info[f'privileged/button_{i}_pos'] = self._data.joint(f'buttonbox_joint_{i}').qpos.copy()
            ob_info[f'privileged/button_{i}_vel'] = self._data.joint(f'buttonbox_joint_{i}').qvel.copy()

        # Drawer states.
        ob_info['privileged/drawer_pos'] = self._data.joint('drawer_slide').qpos.copy()
        ob_info['privileged/drawer_vel'] = self._data.joint('drawer_slide').qvel.copy()
        ob_info['privileged/drawer_handle_pos'] = self._data.site_xpos[self._drawer_site_id].copy()
        ob_info['privileged/drawer_handle_yaw'] = np.array(
            [lie.SO3.from_matrix(self._data.site_xmat[self._drawer_site_id].reshape(3, 3)).compute_yaw_radians()]
        )

        # Window states.
        ob_info['privileged/window_pos'] = self._data.joint('window_slide').qpos.copy()
        ob_info['privileged/window_vel'] = self._data.joint('window_slide').qvel.copy()
        ob_info['privileged/window_handle_pos'] = self._data.site_xpos[self._window_site_id].copy()
        ob_info['privileged/window_handle_yaw'] = np.array(
            [lie.SO3.from_matrix(self._data.site_xmat[self._window_site_id].reshape(3, 3)).compute_yaw_radians()]
        )

        if self._mode == 'data_collection':
            ob_info['privileged/target_task'] = self._target_task

            # Target cube info.
            target_mocap_id = self._cube_target_mocap_ids[self._target_block]
            ob_info['privileged/target_block'] = self._target_block
            ob_info['privileged/target_block_pos'] = self._data.mocap_pos[target_mocap_id].copy()
            ob_info['privileged/target_block_yaw'] = np.array(
                [lie.SO3(wxyz=self._data.mocap_quat[target_mocap_id]).compute_yaw_radians()]
            )

            # Target button info.
            ob_info['privileged/target_button'] = self._target_button
            ob_info['privileged/target_button_state'] = self._target_button_states[self._target_button]
            ob_info['privileged/target_button_top_pos'] = self._data.site_xpos[
                self._button_site_ids[self._target_button]
            ].copy()

            # Target drawer info.
            ob_info['privileged/target_drawer_pos'] = np.array([self._target_drawer_pos])
            ob_info['privileged/target_drawer_handle_pos'] = self._data.site_xpos[self._drawer_target_site_id].copy()

            # Target window info.
            ob_info['privileged/target_window_pos'] = np.array([self._target_window_pos])
            ob_info['privileged/target_window_handle_pos'] = self._data.site_xpos[self._window_target_site_id].copy()

        ob_info['prev_button_states'] = self._prev_button_states.copy()
        ob_info['button_states'] = self._cur_button_states.copy()

    def compute_observation(self):
        if self._ob_type == 'pixels':
            return self.get_pixel_observation()
        else:
            xyz_center = np.array([0.425, 0.0, 0.0])
            xyz_scaler = 10.0
            gripper_scaler = 3.0
            button_scaler = 120.0
            drawer_scaler = 18.0
            window_scaler = 15.0

            ob_info = self.compute_ob_info()
            ob = [
                ob_info['proprio/joint_pos'],
                ob_info['proprio/joint_vel'],
                (ob_info['proprio/effector_pos'] - xyz_center) * xyz_scaler,
                np.cos(ob_info['proprio/effector_yaw']),
                np.sin(ob_info['proprio/effector_yaw']),
                ob_info['proprio/gripper_opening'] * gripper_scaler,
                ob_info['proprio/gripper_contact'],
            ]
            for i in range(self._num_cubes):
                ob.extend(
                    [
                        (ob_info[f'privileged/block_{i}_pos'] - xyz_center) * xyz_scaler,
                        ob_info[f'privileged/block_{i}_quat'],
                        np.cos(ob_info[f'privileged/block_{i}_yaw']),
                        np.sin(ob_info[f'privileged/block_{i}_yaw']),
                    ]
                )
            for i in range(self._num_buttons):
                button_state = np.eye(self._num_button_states)[self._cur_button_states[i]]
                ob.extend(
                    [
                        button_state,
                        ob_info[f'privileged/button_{i}_pos'] * button_scaler,
                        ob_info[f'privileged/button_{i}_vel'],
                    ]
                )
            ob.extend(
                [
                    ob_info['privileged/drawer_pos'] * drawer_scaler,
                    ob_info['privileged/drawer_vel'],
                    ob_info['privileged/window_pos'] * window_scaler,
                    ob_info['privileged/window_vel'],
                ]
            )

            return np.concatenate(ob)
    
    def get_pixel_observation(self):
        camera = 'front_pixels' if not self._multiview else ['front_pixels', 'side_pixels']
        frame = self.render(camera=camera)
        return frame

    def compute_observation(self):
        if self._ob_type == 'pixels':
            return self.get_pixel_observation()
        else:
            xyz_center = np.array([0.425, 0.0, 0.0])
            xyz_scaler = 10.0
            gripper_scaler = 3.0

            ob_info = self.compute_ob_info()
            ob = [
                ob_info['proprio/joint_pos'],
                ob_info['proprio/joint_vel'],
                (ob_info['proprio/effector_pos'] - xyz_center) * xyz_scaler,
                np.cos(ob_info['proprio/effector_yaw']),
                np.sin(ob_info['proprio/effector_yaw']),
                ob_info['proprio/gripper_opening'] * gripper_scaler,
                ob_info['proprio/gripper_contact'],
            ]

            return np.concatenate(ob)

    def compute_oracle_observation(self):
        """Return the oracle goal representation of the current state."""
        xyz_center = np.array([0.425, 0.0, 0.0])
        xyz_scaler = 10.0
        drawer_scaler = 18.0
        window_scaler = 15.0

        ob_info = self.compute_ob_info()
        ob = []
        for i in range(self._num_cubes):
            ob.append((ob_info[f'privileged/block_{i}_pos'] - xyz_center) * xyz_scaler)
        ob.append(self._cur_button_states.astype(np.float64))
        ob.extend(
            [
                ob_info['privileged/drawer_pos'] * drawer_scaler,
                ob_info['privileged/window_pos'] * window_scaler,
            ]
        )

        return np.concatenate(ob)
    
    def compute_reward(self):
        if self._reward_mode == 'sparse':
            reward = 0.0 if self._success else -1.0
        elif self._reward_mode == 'step':
            cube_successes, button_successes, drawer_success, window_success = self._compute_successes()
            successes = cube_successes + button_successes + [drawer_success, window_success]
            reward = float(sum(successes) - len(successes)) / len(successes)
        else:
            raise NotImplementedError(f"Reward mode '{self._reward_mode}' is not implemented.")
        
        return reward
    
    def compute_reward_batch(self, qpos, goal_qpos, button_states, goal_button_states):
        """Compute the reward for a batch of observations and goals from buffer."""
        qpos_obj_start_idx = 14
        qpos_cube_length = 7  # 3 pos + 4 quat
        qpos_drawer_idx = qpos_obj_start_idx + self._num_cubes * qpos_cube_length + self._num_buttons
        qpos_window_idx = qpos_drawer_idx + 1
        
        # Compute successes.
        cube_xyzs_list = []
        goal_xyzs_list = []
        for i in range(self._num_cubes):
            cube_xyzs_list.append(qpos[:, qpos_obj_start_idx + i * qpos_cube_length : qpos_obj_start_idx + i * qpos_cube_length + 3])
            goal_xyzs_list.append(goal_qpos[:, qpos_obj_start_idx + i * qpos_cube_length : qpos_obj_start_idx + i * qpos_cube_length + 3])
        cube_xyzs = torch.stack(cube_xyzs_list, dim=1)
        goal_xyzs = torch.stack(goal_xyzs_list, dim=1)
        cube_successes = torch.linalg.norm(goal_xyzs - cube_xyzs, dim=-1) <= 0.04
        button_successes = button_states == goal_button_states
        drawer_success = torch.abs(qpos[:, qpos_drawer_idx] - goal_qpos[:, qpos_drawer_idx]) <= 0.04
        window_success = torch.abs(qpos[:, qpos_window_idx] - goal_qpos[:, qpos_window_idx]) <= 0.04
        successes = torch.cat([cube_successes, button_successes, drawer_success[:, None], window_success[:, None]], dim=-1)
        
        if self._reward_mode == 'sparse':
            reward = successes.all(dim=-1, keepdim=True).float() - 1
        elif self._reward_mode == 'step':
            reward = (successes.sum(dim=-1) - successes.shape[-1]) / successes.shape[-1]
        else:
            raise NotImplementedError(f"Reward mode '{self._reward_mode}' is not implemented.")

        return reward

    def get_reset_info(self):
        reset_info = self.compute_ob_info()
        if self._mode == 'task':
            reset_info['goal'] = self._cur_goal_ob
            if self._render_goal is not None:
                reset_info['goal_rendered'] = self._cur_goal_rendered
        return reset_info

    def get_step_info(self):
        ob_info = self.compute_ob_info()
        ob_info['success'] = self._success
        ob_info['success_frac'] = self._success_frac
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
