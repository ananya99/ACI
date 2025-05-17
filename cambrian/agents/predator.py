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
        speed: float = 0.9,
        capture_threshold: float = 1.0,
    ):
        super().__init__(config, name)

        self._prey = prey
        self._speed = speed
        self._capture_threshold = capture_threshold
        self.model_path = os.path.join(self.config.model_path, 'agent_predator_model.zip')
        self.extrapolation_fraction = self.config.extrapolation_fraction
        self.model_exists = False
        if os.path.exists(self.model_path):
            self.predator_model = MjCambrianModel.load(self.model_path)
            self.model_exists = True
        else:
            self.predator_model = None
        self.extrapolation_step = 0
        self.delta = 0
        self.prev_action = np.array([-1.0, 0.0])
        self.use_privileged_action = self.config.use_privileged_action

    def reset(self, *args) -> ObsType:
        return super().reset(*args)

    def get_action_privileged(self, env: MjCambrianMazeEnv) -> ActionType:
        assert self._prey in env.agents, f"Prey {self._prey} not found in env"

        if self.use_privileged_action:
            prey_pos = env.agents[self._prey].pos

            target_vector = prey_pos[:2] - self.pos[:2]
            distance = np.linalg.norm(target_vector)

            if distance < self._capture_threshold:
                # get_logger().info(f"{self.name} captured {self._prey}!")
                return [0.0, 0.0]  

            target_theta = np.arctan2(target_vector[1], target_vector[0])
            theta_action = np.interp(target_theta, [-np.pi, np.pi], [-1, 1])

            return [self._speed, theta_action]

        if not self.model_exists:
            if os.path.exists(self.model_path):
                # print(f'[INFO] Loading model of predator from {self.model_path}')
                self.predator_model = MjCambrianModel.load(self.model_path)
                self.model_exists = True
        random_selector = np.random.random()
        if random_selector >= 0.0:
            if self.predator_model is None:
                # print(f'[INFO] Predator Model not found')
                return [-1.0, 0.0]
            obs = env._overlays.get('adversary_obs', False)
            action = self.predator_model.predict(obs, deterministic=True)
            action = action[0]
            self.delta = action - self.prev_action
            self.prev_action = action
            self.extrapolation_step = 0
            return action
        else:
            self.extrapolation_step += 1
            return self.prev_action + self.extrapolation_step * self.delta
