"""Wrappers for the MjCambrianEnv. Used during training."""

from pathlib import Path
from types import NoneType
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from cambrian.ml.model import MjCambrianModel
import gymnasium as gym
import numpy as np
from gymnasium.wrappers.numpy_to_torch import numpy_to_torch, torch_to_numpy
from stable_baselines3.common.env_checker import check_env
from gymnasium.wrappers import FrameStackObservation
from itertools import cycle

from cambrian.envs import MjCambrianEnv, MjCambrianEnvConfig
from cambrian.utils import device, is_integer
from cambrian.utils.types import (
    ActionType,
    InfoType,
    ObsType,
    RenderFrame,
    RewardType,
    TerminatedType,
    TruncatedType,
)


class MjCambrianSingleAgentEnvWrapper(gym.Wrapper):
    """Wrapper around the MjCambrianEnv that acts as if there is a single agent.

    Will replace all multi-agent methods to just use the first agent.

    Keyword Args:
        agent_name: The name of the agent to use. If not provided, the first agent
            will be used.
    """

    def __init__(
        self,
        env: MjCambrianEnv,
        *,
        agent_name: Optional[str] = None,
        combine_rewards: bool = True,
        combine_terminated: bool = True,
        combine_truncated: bool = True,
    ):
        super().__init__(env)

        self._combine_rewards = combine_rewards
        self._combine_terminated = combine_terminated
        self._combine_truncated = combine_truncated

        agent_name = agent_name or next(iter(env.agents.keys()))
        assert agent_name in env.agents, f"agent {agent_name} not found."
        self._agent = env.agents[agent_name]
        self.action_space = self._agent.action_space
        self.observation_space = self._agent.observation_space

    def reset(self, *args, **kwargs) -> Tuple[ObsType, InfoType]:
        obs, info = self.env.reset(*args, **kwargs)

        return obs[self._agent.name], info[self._agent.name]

    def step(
        self, action: ActionType
    ) -> Tuple[ObsType, RewardType, TerminatedType, TruncatedType, InfoType]:
        action = {self._agent.name: action}
        obs, reward, terminated, truncated, info = self.env.step(action)

        obs = obs[self._agent.name]
        info = info[self._agent.name]

        if self._combine_rewards:
            reward = sum(list(reward.values()))
        else:
            reward = reward[self._agent.name]

        if self._combine_terminated:
            terminated = any(terminated.values())
        else:
            terminated = terminated[self._agent.name]

        if self._combine_truncated:
            truncated = any(truncated.values())
        else:
            truncated = truncated[self._agent.name]

        return obs, reward, terminated, truncated, info
    
class MjCambrianAlternateTrainingEnvWrapper(gym.Wrapper):
    """Wrapper around the MjCambrianEnv that acts as if there is a single agent.

    Will replace all multi-agent methods to just use the first agent.

    Keyword Args:
        agent_name: The name of the agent to use. If not provided, the first agent
            will be used.
    """

    def __init__(
        self,
        env: MjCambrianEnv,
        *,
        pretrained_policy_path: str = None,
        agent_name: Optional[str] = None,
        combine_rewards: bool = True,
        combine_terminated: bool = True,
        combine_truncated: bool = True,
    ):
        super().__init__(env)

        self.pretrained_policy_path = pretrained_policy_path
        # print("pretrained_policy_path: ", pretrained_policy_path)
        self._combine_rewards = combine_rewards
        self._combine_terminated = combine_terminated
        self._combine_truncated = combine_truncated

        agent_name = list(env.agents.keys())[1]
        print("initial agent_name: ", agent_name)
        assert agent_name in env.agents, f"agent {agent_name} not found."
        
        self._training_agent = env.agents[agent_name]
        self.action_space = self._training_agent.action_space
        self.observation_space = self._training_agent.observation_space
        
        self.last_obs = None
        self._agent_models = env.agent_models
        self.prev_actions = np.array([[-1.0, 0.0] for _ in range(len(env.agents))])
        self._agent_model_config = None
        
    def set_agent_model_config(self, agent_model_config):
        print("setting agent_model_config")
        self._agent_model_config = agent_model_config

    # def set_agent_models(self, agent_models):
    #     print("setting agent models: ", agent_models)
    #     self.agent_models = agent_models

    # def set_training_agent(self, agent_name):
    #     self._training_agent = self.env.agents[agent_name]

    def is_training_agent(self, agent_name):
        # print("is_training_agent: ", agent_name, self._training_agent.name)
        return agent_name == self._training_agent.name

    def reset(self, *args, **kwargs) -> Tuple[ObsType, InfoType]:
        obs, info = self.env.reset(*args, **kwargs)
        return obs[self._training_agent.name], info[self._training_agent.name]
    
    def fill_action(self,i,agent_name, training_agent_action):
        if self.is_training_agent(agent_name):
            print("it's the training agent, returning action")
            self.prev_actions[i] = training_agent_action
            return training_agent_action
        else:
            # return self.prev_actions[i]
            # load the pretrained policy from the policy.pt file
            if self._agent_model_config is None:
                print("model config is not set.")
                return self.prev_actions[i]
            policy_path = Path(self.pretrained_policy_path) / f"{agent_name}_policy.pt"
            print("policy_path: ", policy_path)
            if not policy_path.exists():
                raise FileNotFoundError(f"Could not find pretrained policy for {agent_name}_policy.pt file at {policy_path}.")
            model = self._agent_model_config.model()
            pretrained_policy = model.load_policy(policy_path)
            print("loaded pretrained_policy: ", pretrained_policy)
            action, _ = pretrained_policy.predict(self.last_obs[agent_name])
            self.prev_actions[i] = action
            return action

    def step(
        self, action: ActionType
    ) -> Tuple[ObsType, RewardType, TerminatedType, TruncatedType, InfoType]:
        # actions = {self._training_agent.name: action}
        # fill actions for all the other agents
        training_agent_action = action
        actions = {
            agent_name: self.fill_action(i,agent_name, training_agent_action)
            for i, agent_name in enumerate(self.env.agents.keys())
            if self.env.agents[agent_name].config.trainable
        }
        
        # print("actions: ", actions)
        
        obs, rewards, terminateds, truncateds, infos = self.env.step(actions)
        self.last_obs = obs

        ob = obs[self._training_agent.name]
        info = infos[self._training_agent.name]
        
        reward = rewards[self._training_agent.name]

        if self._combine_terminated:
            terminated = any(terminateds.values())
        else:
            terminated = terminateds[self._training_agent.name]

        if self._combine_truncated:
            truncated = any(truncateds.values())
        else:
            truncated = truncateds[self._training_agent.name]

        return ob, reward, terminated, truncated, info

class MjCambrianAECEnvWrapper(gym.Wrapper):
    def __init__(self, env: MjCambrianEnv):
        super().__init__(env)
        self.env: MjCambrianEnv
        self.agents = cycle(env.agents)
        self.selected_agent = None
        self.prev_action = np.array([[-1.0,0.0] for _ in range(len(env.agents))])

    def check_agent_selection(self,agent_name):
        return agent_name == self.selected_agent

    def obs_mask(self, obj):
        if isinstance(obj,list) :
            for i, x in enumerate(obj):
                obj[i] = x*0
        else:
            obj = obj * 0
        return obj

    def action_mask(self, action, i , agent_name):
        if self.check_agent_selection(agent_name):
            self.prev_action[:,i] = action[:,i]
            return action[:,i]
        else:
            return self.prev_action[:,i]

    def iter_agent(self):
        self.selected_agent = next(self.agents)

    def reset(self, *args, **kwargs) -> Tuple[ObsType, InfoType]:
        obs, info = self.env.reset(*args, **kwargs)
        # Flatten the observations
        self.iter_agent()
        flattened_obs: Dict[str, Any] = {}
        for agent_name, agent_obs in obs.items():
            if isinstance(agent_obs, dict):
                for key, value in agent_obs.items():
                    flattened_obs[f"{agent_name}_{key}"] = value if self.check_agent_selection(agent_name) else self.obs_mask(value)
            else:
                flattened_obs[agent_name] = agent_obs if self.check_agent_selection(agent_name) else self.obs_mask(agent_obs)
        return flattened_obs, info

    def step(
        self, action: ActionType
    ) -> Tuple[ObsType, RewardType, TerminatedType, TruncatedType, InfoType]:
        # Convert the action back to a dict
        action = action.reshape(-1, len(self.env.agents))
        action = {
            agent_name: self.action_mask(action,i,agent_name)
            for i, agent_name in enumerate(self.env.agents.keys())
            if self.env.agents[agent_name].config.trainable
        }

        obs, reward, terminated, truncated, info = self.env.step(action)

        self.iter_agent()

        # Accumulate the rewards, terminated, and truncated
        reward = reward[self.selected_agent]
        terminated = terminated[self.selected_agent]
        truncated = truncated[self.selected_agent]
        
        # print("reward: ", reward, "agent: ", self.selected_agent)

        # Flatten the observations
        flattened_obs: Dict[str, Any] = {}
        for agent_name, agent_obs in obs.items():
            if isinstance(agent_obs, dict):
                for key, value in agent_obs.items():
                    flattened_obs[f"{agent_name}_{key}"] = value if self.check_agent_selection(agent_name) else self.obs_mask(value)
            else:
                flattened_obs[agent_name] = agent_obs if self.check_agent_selection(agent_name) else self.obs_mask(agent_obs)

        return flattened_obs, reward, terminated, truncated, info

    @property
    def observation_space(self) -> gym.spaces.Dict:
        """SB3 doesn't support nested Dict observation spaces, so we'll flatten it.
        If each agent has a Dict observation space, we'll flatten it into a single
        observation where the key in the dict is the agent name and the original space
        name."""
        observation_space: Dict[str, gym.Space] = {}
        for agent in self.env.agents.values():
            agent_observation_space = agent.observation_space
            if isinstance(agent_observation_space, gym.spaces.Dict):
                for key, value in agent_observation_space.spaces.items():
                    observation_space[f"{agent.name}_{key}"] = value
            else:
                observation_space[agent.name] = agent_observation_space
        return gym.spaces.Dict(observation_space)

    @property
    def action_space(self) -> gym.spaces.Box:
        """The only gym.Space that SB3 supports that's continuous for the action space
        is a Box. We can assume each agent's action space is a Box, so we'll flatten
        each action space into one Box for the environment.

        Assumptions:
            - All agents have the same number of actions
            - All actions have the same shape
            - All actions are continuous
            - All actions are normalized between -1 and 1
        """

        # Get the first agent's action space
        first_agent_name = next(iter(self.env.agents.keys()))
        first_agent_action_space = self.env.agents[first_agent_name].action_space

        # Check if the action space is continuous
        assert isinstance(first_agent_action_space, gym.spaces.Box), (
            "SB3 only supports continuous action spaces for the environment. "
            f"agent {first_agent_name} has a {type(first_agent_action_space)}"
            " action space."
        )

        # Get the shape of the action space
        shape = first_agent_action_space.shape
        low = first_agent_action_space.low
        high = first_agent_action_space.high

        # Check if all agents have the same number of actions
        for agent_name, agent_action_space in self.env.action_spaces.items():
            assert shape == agent_action_space.shape, (
                "All agents must have the same number of actions. "
                f"agent {first_agent_name} has {shape} actions, but {agent_name} "
                f"has {agent_action_space.shape} actions."
            )

            # Check if the action space is continuous
            assert isinstance(agent_action_space, gym.spaces.Box), (
                "SB3 only supports continuous action spaces for the environment. "
                f"agent {first_agent_name} has a "
                f"{type(first_agent_action_space)} action space."
            )

            assert all(low == agent_action_space.low), (
                "All actions must have the same low value. "
                f"agent {first_agent_name} has a low value of {low}, "
                f"but {agent_name} has a low value of {agent_action_space.low}."
            )

            assert all(high == agent_action_space.high), (
                "All actions must have the same high value. "
                f"agent {first_agent_name} has a high value of {high}, "
                f"but {agent_name} has a high value of {agent_action_space.high}."
            )

        low = np.tile(low, len(self.env.agents))
        high = np.tile(high, len(self.env.agents))
        shape = (shape[0] * len(self.env.agents),)
        return gym.spaces.Box(
            low=low, high=high, shape=shape, dtype=first_agent_action_space.dtype
        )
    

class MjCambrianPettingZooEnvWrapper(gym.Wrapper):
    """Wrapper around the MjCambrianEnv that acts as if there is a single agent, where
    in actuality, there's multi-agents.

    SB3 doesn't support Dict action spaces, so this wrapper will flatten the action
    into a single space. The observation can be a dict; however, nested dicts are not
    allowed.
    """

    def __init__(self, env: MjCambrianEnv):
        super().__init__(env)
        self.env: MjCambrianEnv

    def reset(self, *args, **kwargs) -> Tuple[ObsType, InfoType]:
        obs, info = self.env.reset(*args, **kwargs)

        # Flatten the observations
        flattened_obs: Dict[str, Any] = {}
        for agent_name, agent_obs in obs.items():
            if isinstance(agent_obs, dict):
                for key, value in agent_obs.items():
                    flattened_obs[f"{agent_name}_{key}"] = value
            else:
                flattened_obs[agent_name] = agent_obs

        return flattened_obs, info

    def step(
        self, action: ActionType
    ) -> Tuple[ObsType, RewardType, TerminatedType, TruncatedType, InfoType]:
        # Convert the action back to a dict
        action = action.reshape(-1, len(self.env.agents))
        action = {
            agent_name: action[:, i]
            for i, agent_name in enumerate(self.env.agents.keys())
            if self.env.agents[agent_name].config.trainable
        }

        obs, reward, terminated, truncated, info = self.env.step(action)

        # Accumulate the rewards, terminated, and truncated
        reward = sum(reward.values())
        terminated = any(terminated.values())
        truncated = any(truncated.values())

        # Flatten the observations
        flattened_obs: Dict[str, Any] = {}
        for agent_name, agent_obs in obs.items():
            if isinstance(agent_obs, dict):
                for key, value in agent_obs.items():
                    flattened_obs[f"{agent_name}_{key}"] = value
            else:
                flattened_obs[agent_name] = agent_obs

        return flattened_obs, reward, terminated, truncated, info

    @property
    def observation_space(self) -> gym.spaces.Dict:
        """SB3 doesn't support nested Dict observation spaces, so we'll flatten it.
        If each agent has a Dict observation space, we'll flatten it into a single
        observation where the key in the dict is the agent name and the original space
        name."""
        observation_space: Dict[str, gym.Space] = {}
        for agent in self.env.agents.values():
            agent_observation_space = agent.observation_space
            if isinstance(agent_observation_space, gym.spaces.Dict):
                for key, value in agent_observation_space.spaces.items():
                    observation_space[f"{agent.name}_{key}"] = value
            else:
                observation_space[agent.name] = agent_observation_space
        return gym.spaces.Dict(observation_space)

    @property
    def action_space(self) -> gym.spaces.Box:
        """The only gym.Space that SB3 supports that's continuous for the action space
        is a Box. We can assume each agent's action space is a Box, so we'll flatten
        each action space into one Box for the environment.

        Assumptions:
            - All agents have the same number of actions
            - All actions have the same shape
            - All actions are continuous
            - All actions are normalized between -1 and 1
        """

        # Get the first agent's action space
        first_agent_name = next(iter(self.env.agents.keys()))
        first_agent_action_space = self.env.agents[first_agent_name].action_space

        # Check if the action space is continuous
        assert isinstance(first_agent_action_space, gym.spaces.Box), (
            "SB3 only supports continuous action spaces for the environment. "
            f"agent {first_agent_name} has a {type(first_agent_action_space)}"
            " action space."
        )

        # Get the shape of the action space
        shape = first_agent_action_space.shape
        low = first_agent_action_space.low
        high = first_agent_action_space.high

        # Check if all agents have the same number of actions
        for agent_name, agent_action_space in self.env.action_spaces.items():
            assert shape == agent_action_space.shape, (
                "All agents must have the same number of actions. "
                f"agent {first_agent_name} has {shape} actions, but {agent_name} "
                f"has {agent_action_space.shape} actions."
            )

            # Check if the action space is continuous
            assert isinstance(agent_action_space, gym.spaces.Box), (
                "SB3 only supports continuous action spaces for the environment. "
                f"agent {first_agent_name} has a "
                f"{type(first_agent_action_space)} action space."
            )

            assert all(low == agent_action_space.low), (
                "All actions must have the same low value. "
                f"agent {first_agent_name} has a low value of {low}, "
                f"but {agent_name} has a low value of {agent_action_space.low}."
            )

            assert all(high == agent_action_space.high), (
                "All actions must have the same high value. "
                f"agent {first_agent_name} has a high value of {high}, "
                f"but {agent_name} has a high value of {agent_action_space.high}."
            )

        low = np.tile(low, len(self.env.agents))
        high = np.tile(high, len(self.env.agents))
        shape = (shape[0] * len(self.env.agents),)
        return gym.spaces.Box(
            low=low, high=high, shape=shape, dtype=first_agent_action_space.dtype
        )


class MjCambrianConstantActionWrapper(gym.Wrapper):
    """This wrapper will apply a constant action at specific indices of the action
    space.

    Args:
        constant_actions: A dictionary where the keys are the indices of the action
            space and the values are the constant actions to apply.
    """

    def __init__(self, env: MjCambrianEnv, constant_actions: Dict[Any, Any]):
        super().__init__(env)

        self._constant_action_indices = [
            int(k) if is_integer(k) else k for k in constant_actions.keys()
        ]
        self._constant_action_values = list(constant_actions.values())

    def step(
        self, action: ActionType
    ) -> Tuple[ObsType, RewardType, TerminatedType, TruncatedType, InfoType]:
        if isinstance(action, dict):
            assert all(idx in action for idx in self._constant_action_indices), (
                "The constant action indices must be in the action space."
                f"Indices: {self._constant_action_indices}, Action space: {action}"
            )
        action[self._constant_action_indices] = self._constant_action_values

        return self.env.step(action)
    
    def set_agent_model_config(self, agent_model_config):
        self.env.set_agent_model_config(agent_model_config)


@torch_to_numpy.register(np.ndarray)
def _(value: np.ndarray) -> np.ndarray:
    return value


@torch_to_numpy.register(NoneType)
def _(value: NoneType) -> NoneType:
    return value


class MjCambrianTorchToNumpyWrapper(gym.Wrapper):
    """Wraps a torch-based environment to convert inputs and outputs to NumPy arrays."""

    def __init__(self, env: gym.Env, *, convert_action: bool = False):
        """Wrapper class to change inputs and outputs of environment to numpy arrays.

        Args:
            env: The torch-based environment

        Keyword Args:
            convert_action: Whether to convert the action to a numpy array
        """
        super().__init__(env)

        self._convert_action = convert_action

    def step(
        self, actions: ActionType
    ) -> Tuple[ObsType, RewardType, TerminatedType, TruncatedType, InfoType]:
        """Using a numpy-based action that is converted to torch to be used by the
        environment.

        Args:
            action: A numpy-based action

        Returns:
            The numpy-based observation, reward, termination, truncation, and extra info
        """
        actions = (
            numpy_to_torch(actions, device=device) if self._convert_action else actions
        )
        obs, reward, terminated, truncated, info = self.env.step(actions)

        return (
            torch_to_numpy(obs),
            reward,
            terminated,
            truncated,
            torch_to_numpy(info),
        )

    def reset(
        self,
        *,
        seed: int | list[int] | None = None,
        options: dict[str, Any] | None = None,
    ) -> Tuple[ObsType, InfoType]:
        """Resets the environment returning numpy-based observations and info.

        Args:
            seed: The seed for resetting the environment
            options: The options for resetting the environment

        Returns:
            The numpy-based observation and extra info
        """
        if options:
            options = numpy_to_torch(options, device=device)

        obs, info = self.env.reset(seed=seed, options=options)
        return torch_to_numpy(obs), torch_to_numpy(info)

    def render(self) -> RenderFrame | List[RenderFrame] | None:
        """Renders the environment returning a numpy-based image.

        Returns:
            The numpy-based image
        """
        return torch_to_numpy(self.env.render())
    
    def set_agent_model_config(self, agent_model_config):
        self.env.set_agent_model_config(agent_model_config)
    
# Add a wrapper equivalent of gymnasium.wrappers.FrameStackObservation
class MjCambrianFrameStackWrapper(FrameStackObservation):
    def __init__(self,env: gym.Env, stack_size: int, *, padding_type: str | ObsType = "reset"):
        super().__init__(env, stack_size, padding_type=padding_type)
        
    def reset(self, *args, **kwargs) -> Tuple[ObsType, InfoType]:
        obs, info = super().reset(*args, **kwargs)
        return obs, info
    
    def step(self, action: ActionType) -> Tuple[ObsType, RewardType, TerminatedType, TruncatedType, InfoType]:
        obs, reward, terminated, truncated, info = super().step(action)
        return obs, reward, terminated, truncated, info
    
    def set_agent_model_config(self, agent_model_config):
        self.env.set_agent_model_config(agent_model_config)


def make_wrapped_env(
    config: MjCambrianEnvConfig,
    wrappers: List[Callable[[gym.Env], gym.Env]],
    seed: Optional[int] = None,
    agent_model_config = None,
    **kwargs,
) -> gym.Env:
    """Utility function for creating a MjCambrianEnv."""

    def _init():
        env = config.instance(config, **kwargs)
        for wrapper in wrappers:
            # print(f"Wrapping {env} with {wrapper}")
            env = wrapper(env)
            if hasattr(env, 'set_agent_model_config') and agent_model_config is not None:
                env.set_agent_model_config(agent_model_config)
            # if isinstance(wrapper, MjCambrianMultiAgentEnvWrapper):
            #     print("cambrian_env.agents.keys: ", env.agents.keys())
        # check_env will call reset and set the seed to 0; call set_random_seed after
        # print(f"Checking env {env}")
        # print("observation space:", env.observation_space)
        # print("action space:", env.action_space)
        # print("cambrian_env.agents.keys: ", env.agents.keys())
        check_env(env, warn=False)
        env.unwrapped.set_random_seed(seed)
        return env

    return _init
