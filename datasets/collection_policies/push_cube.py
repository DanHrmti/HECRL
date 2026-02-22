import numpy as np

from ogbench.manipspace.oracles.markov.markov_oracle import MarkovOracle


class PushCubeMarkovOracle(MarkovOracle):
    def __init__(self, max_step=50, max_push_step=10, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._max_step = max_step
        self._max_push_step = max_push_step
        self._face_to_coor = {"right": [1, 1], "left": [-1, -1], "front": [-1, 1], "back": [1, -1]}

    def reset(self, ob, info):
        self._done = False
        self._step = 0
        self._push_step = 0
        self._push_position_reached = False
        self._target_face = np.random.choice(['right', 'left', 'front', 'back'], p=[0.25, 0.25, 0.25, 0.25])

    def select_action(self, ob, info):
        effector_pos = info['proprio/effector_pos']
        effector_yaw = info['proprio/effector_yaw'][0]

        target_block = info['privileged/target_block']
        block_pos = info[f'privileged/block_{target_block}_pos']
        block_yaw = self.shortest_yaw(effector_yaw, info[f'privileged/block_{target_block}_yaw'][0])

        block_above_offset = np.array([0, 0, 0.18])
        block_side_offset = np.array([0.07 * self._face_to_coor[self._target_face][0] * np.cos(block_yaw),
                                      0.07 * self._face_to_coor[self._target_face][1] * np.sin(block_yaw),
                                      0.0])
        xy_aligned = np.linalg.norm(block_pos[:2] + block_side_offset[:2] - effector_pos[:2]) <= 0.04
        pos_aligned = np.linalg.norm(block_pos + block_side_offset - effector_pos) <= 0.02

        gain_pos = 5
        gain_yaw = 3
        action = np.zeros(5)
        if not self._push_position_reached:
            if not xy_aligned:
                self.print_phase('1: Move above a side of the block')
                action = np.zeros(5)
                diff = block_pos + block_above_offset + block_side_offset - effector_pos
                diff = self.shape_diff(diff)
                action[:3] = diff[:3] * gain_pos
                action[3] = (block_yaw - effector_yaw) * gain_yaw
                action[4] = -1
            elif not pos_aligned:
                self.print_phase('2: Move to a side of the block')
                diff = block_pos + block_side_offset - effector_pos
                diff = self.shape_diff(diff)
                action[:3] = diff[:3] * gain_pos
                action[3] = (block_yaw - effector_yaw) * gain_yaw
                action[4] = 1
            else:
                self._push_position_reached = True

        if self._push_position_reached:
            self.print_phase('3: Move towards the block (push)')
            diff = block_pos - effector_pos
            diff = self.shape_diff(diff)
            action[:3] = diff[:3] * gain_pos
            action[3] = (block_yaw - effector_yaw) * gain_yaw
            action[4] = 1
            self._push_step += 1

        action = np.clip(action, -1, 1)
        if self._debug:
            print(action)

        self._step += 1
        if self._step == self._max_step or self._push_step == self._max_push_step:
            self._done = True

        return action
