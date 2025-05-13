from typing import Optional
import numpy as np
from cambrian.agents.agent import MjCambrianAgentConfig
from cambrian.envs.maze_env import MjCambrianMazeEnv
from cambrian.ml.model import MjCambrianModel
from cambrian.utils import get_logger
from cambrian.utils.types import ActionType, ObsType
from .point import MjCambrianAgentPoint
import os

class MjCambrianAgentPredator(MjCambrianAgentPoint):

    def __init__(
        self,
        config: MjCambrianAgentConfig,
        name: str,
        *,
        prey: str,
        speed: float = 1.0,
        capture_threshold: float = 1.0,
    ):
        super().__init__(config, name)

        self._prey = prey
        self._speed = speed
        self._capture_threshold = capture_threshold
        self.model_path = os.path.join(self.config.model_path, 'agent_predator_model.zip')
        if os.path.exists(self.model_path):
            self.predator_model = MjCambrianModel.load(self.model_path)
        else:
            self.predator_model = None

    def reset(self, *args) -> ObsType:
        return super().reset(*args)

    def get_action_privileged(self, env: MjCambrianMazeEnv) -> ActionType:
        assert self._prey in env.agents, f"Prey {self._prey} not found in env"
        # obs = env._overlays.get('adversary_obs',False)
        # if np.random.random() > 0.5:
        #     if obs:
        #         action = self.predator_model.predict(obs, deterministic=True)
        #         action = action[0]
        #         return action
        if os.path.exists(self.model_path):
            self.predator_model = MjCambrianModel.load(self.model_path)
        obs = env._overlays.get('adversary_obs',False)
        # This is for check to work as the model won't exist at that time
        if self.predator_model is None:
            print(f'Model not found')
            return [-1.0, 0.0]
        action = self.predator_model.predict(obs, deterministic=True)
        action = action[0]
        # self.delta = action - self.prev_action
        # self.prev_action = action
        # self.extrapolation_step = 0
        return action

        # target_vector = prey_pos[:2] - self.pos[:2]
        # distance = np.linalg.norm(target_vector)

        # if distance < self._capture_threshold:
        #     # get_logger().info(f"{self.name} captured {self._prey}!")
        #     return [0.0, 0.0]

        # target_theta = np.arctan2(target_vector[1], target_vector[0])
        # theta_action = np.interp(target_theta, [-np.pi, np.pi], [-1, 1])

        # return [self._speed, theta_action]