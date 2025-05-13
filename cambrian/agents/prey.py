from typing import Optional
import numpy as np
from cambrian.agents.agent import MjCambrianAgentConfig
from cambrian.envs.maze_env import MjCambrianMazeEnv
from cambrian.utils import get_logger
from cambrian.utils.types import ActionType, ObsType
from .point import MjCambrianAgentPoint
from cambrian.ml.model import MjCambrianModel


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
        self.prey_model = MjCambrianModel.load('/home/neo/Projects/vi/project/ACI/logs/2025-05-13/exp_detection/prey_model.zip')
        self.action_buffer = [-1.0,0.0]

    def reset(self, *args) -> ObsType:
        return super().reset(*args)

    def get_action_privileged(self, env: MjCambrianMazeEnv) -> ActionType:
        assert self._predator in env.agents, f"Predator {self._predator} not found in env"

        if(np.random.random() > 0.5):
            obs = env._overlays.get('adversary_obs',False)
            if obs:
                action = self.prey_model.predict(obs, deterministic=True)
                action = action[0].reshape(-1, len(env.agents))
                self.action_buffer = action
                return action[:,0]
        else:
            return self.action_buffer

        predator_pos = env.agents[self._predator].pos

        escape_vector = self.pos[:2] - predator_pos[:2]
        distance = np.linalg.norm(escape_vector)

        if distance > self._safe_distance:
            # get_logger().info(f"{self.name} is safe from {self._predator}.")
            return [-1.0, 0.0]

        escape_theta = np.arctan2(escape_vector[1], escape_vector[0])
        theta_action = np.interp(escape_theta, [-np.pi, np.pi], [-1, 1])

        return [self._speed, theta_action]