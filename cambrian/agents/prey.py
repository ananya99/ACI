from typing import Optional
import numpy as np
from cambrian.agents.agent import MjCambrianAgentConfig
from cambrian.envs.maze_env import MjCambrianMazeEnv
from cambrian.utils import get_logger
from cambrian.utils.types import ActionType, ObsType
from .point import MjCambrianAgentPoint
from cambrian.ml.model import MjCambrianModel
import os

class MjCambrianAgentPrey(MjCambrianAgentPoint):

    def __init__(
        self,
        config: MjCambrianAgentConfig,
        name: str,
        *,
        predator: str,
        speed: float = 1.0,
        safe_distance: float = 5.0,
    ):
        super().__init__(config, name)

        self._predator = predator
        self._speed = speed
        self._safe_distance = safe_distance
        self.model_path = os.path.join(self.config.model_path, 'agent_prey_model.zip')
        self.model_exists = False
        if os.path.exists(self.model_path):
            self.prey_model = MjCambrianModel.load(self.model_path)
            self.model_exists = True
        else:
            self.prey_model = None
        self.extrapolation_step = 0
        self.delta = 0
        self.prev_action = np.array([-1.0,0.0])

    def reset(self, *args) -> ObsType:
        return super().reset(*args)

    def get_action_privileged(self, env: MjCambrianMazeEnv) -> ActionType:
        assert self._predator in env.agents, f"Predator {self._predator} not found in env"
        if not self.model_exists:
            if os.path.exists(self.model_path):
                print(f'[INFO] Prey Model is {self.prey_model}, need to load again')
                print(f'[INFO] Loading model of prey from {self.model_path}')
                self.prey_model = MjCambrianModel.load(self.model_path)
                self.model_exists = True
        if self.prey_model is None:
            # print(f'Prey Model not found')
            return [-1.0, 0.0]
        # random_selector = np.random.random()
        # if random_selector > 0.75:
        obs = env._overlays.get('adversary_obs', False)
        action = self.prey_model.predict(obs, deterministic=True)
        action = action[0]
        # self.delta = action - self.prev_action
        # self.prev_action = action
        # self.extrapolation_step = 0
        return action
        # elif random_selector > 0.25:
        #     self.extrapolation_step += 1
        #     return self.prev_action + self.extrapolation_step * self.delta

        # predator_pos = env.agents[self._predator].pos

        # escape_vector = self.pos[:2] - predator_pos[:2]
        # distance = np.linalg.norm(escape_vector)

        # if distance > self._safe_distance:
        #     # get_logger().info(f"{self.name} is safe from {self._predator}.")
        #     return [-1.0, 0.0]

        # escape_theta = np.arctan2(escape_vector[1], escape_vector[0])
        # theta_action = np.interp(escape_theta, [-np.pi, np.pi], [-1, 1])

        # return [self._speed, theta_action]