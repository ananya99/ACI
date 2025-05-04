"""This module contains the trainer class for training and evaluating agents."""

from pathlib import Path
from typing import TYPE_CHECKING, Callable, Concatenate, Dict, Optional, Any

from hydra_config import HydraContainerConfig, config_wrapper
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecEnv,
    VecMonitor,
)

import gymnasium as gym

from cambrian.envs.env import MjCambrianEnv, MjCambrianEnvConfig
from cambrian.ml.model import MjCambrianModel
from cambrian.utils import evaluate_policy
from cambrian.utils.logger import get_logger
from cambrian.utils.wrappers import make_wrapped_env, MjCambrianMultiAgentEnvWrapper
from dataclasses import field # Import field

from ray.rllib.algorithms.ppo import PPOConfig # Example: Using PPO
from ray.rllib.policy.policy import PolicySpec
from ray.tune.registry import register_env
import ray

if TYPE_CHECKING:
    from cambrian import MjCambrianConfig


@config_wrapper
class MjCambrianTrainerConfig(HydraContainerConfig):
    """Settings for the training process. Used for type hinting.

    Attributes:
        total_timesteps (int): The total number of timesteps to train for.
        max_episode_steps (int): The maximum number of steps per episode.
        n_envs (int): The number of parallel environments to use for training.

        model (Callable[[MjCambrianEnv], MjCambrianModel]): The model to use for
            training.
        callbacks (Dict[str, BaseCallback]): The callbacks to use for training.
        wrappers (Dict[str, Callable[[VecEnv], VecEnv]] | None): The wrappers to use for
            training. If None, will ignore.

        prune_fn (Optional[Callable[[MjCambrianConfig], bool]]): The function to use to
            determine if an experiment should be pruned. If None, will ignore. If set,
            this function will be called prior to training to check whether the config
            is valid for training. This is the get around the fact that some sweepers
            will evaluate configs that are invalid for training, which is a waste
            computationally. The train method will return -inf if this function returns
            True. NOTE: for nevergrad, it is recommended to use cheap_constraints.
        fitness_fn (Callable[[MjCambrianConfig, float]]): The function to use to
            calculate the fitness of the agent after training.
    """

    total_timesteps: int
    max_episode_steps: int
    n_envs: int

    model: Callable[[MjCambrianEnv], MjCambrianModel]
    callbacks: Dict[str, BaseCallback | Callable[[VecEnv], BaseCallback]]
    wrappers: Dict[str, Callable[[VecEnv], VecEnv] | None]

    prune_fn: Optional[Callable[[Concatenate["MjCambrianConfig", ...]], bool]] = None
    fitness_fn: Callable[Concatenate["MjCambrianConfig", ...], float]

    rllib_algorithm: str = "PPO"
    rllib_config_updates: Dict[str, Any] = field(default_factory=dict)
    num_rollout_workers: int = 1 


class MjCambrianTrainer:
    """This is the trainer class for running training and evaluation.

    Args:
        config (MjCambrianConfig): The config to use for training and evaluation.
    """

    def __init__(self, config: "MjCambrianConfig"):
        self._config = config

        self._config.expdir.mkdir(parents=True, exist_ok=True)    
        get_logger().info(f"Logging to {self._config.expdir / 'logs'}...")
        # Initialize Ray
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True, log_to_driver=False) # Configure logging as needed

    def train(self) -> float:
        """Train the agent."""

        # Set to warn so we have something output to the error log
        get_logger().warning(f"Training the agent in {self._config.expdir}...")

        self._config.save(self._config.expdir / "config.yaml")

        # Delete an existing finished file, if it exists
        if (finished := self._config.expdir / "finished").exists():
            finished.unlink()

        # Prune the experiment, if necessary
        if (prune_fn := self._config.trainer.prune_fn) and prune_fn(self._config):
            Path(self._config.expdir / "pruned").touch()
            return -float("inf")

        # Setup the environment, model, and callbacks
        env = self._make_env(self._config.env, self._config.trainer.n_envs)
        eval_env = self._make_env(self._config.eval_env, 1, monitor="eval_monitor.csv")
        callback = self._make_callback(eval_env)
        model = self._make_model(env)

        # Save the eval environments xml
        cambrian_env: MjCambrianEnv = eval_env.envs[0].unwrapped
        cambrian_env.xml.write(self._config.expdir / "env.xml")
        with open(self._config.expdir / "compiled_env.xml", "w") as f:
            f.write(cambrian_env.spec.to_xml())

        # Start training
        total_timesteps = self._config.trainer.total_timesteps
        model.learn(total_timesteps=total_timesteps, callback=callback)
        get_logger().info("Finished training the agent...")

        # Save the policy
        get_logger().info(f"Saving model to {self._config.expdir}...")
        model.save_policy(self._config.expdir)
        get_logger().debug(f"Saved model to {self._config.expdir}...")

        # The finished file indicates to the evo script that the agent is done
        Path(self._config.expdir / "finished").touch()

        # Calculate fitness
        fitness = self._config.trainer.fitness_fn(self._config)
        get_logger().info(f"Final Fitness: {fitness}")

        # Save the final fitness to a file
        with open(self._config.expdir / "train_fitness.txt", "w") as f:
            f.write(str(fitness))

        return fitness

    def eval(
        self,
        *,
        filename: Optional[Path | str] = None,
        record: bool = True,
        load_if_exists: bool = False,
        **callback_kwargs,
    ) -> float:
        self._config.save(self._config.expdir / "eval_config.yaml")
        eval_env_name = "cambrian_eval_env"
        self._register_env(eval_env_name, self._config.eval_env)

        algo = self._setup_rllib_algorithm(env_name=eval_env_name) # Use eval env config? Check setup

        checkpoint_dir = self._config.expdir

        if load_if_exists and Path(checkpoint_dir).exists(): # Check if dir exists
             try:
                 algo.restore(checkpoint_dir) # Try restoring from the directory
                 get_logger().info(f"Loaded model from checkpoint: {checkpoint_dir}")
             except Exception as e:
                 get_logger().error(f"Could not restore checkpoint from {checkpoint_dir}: {e}")
                 return -float("inf") # Or raise error
        else:
             get_logger().warning("No checkpoint found or load_if_exists=False. Evaluating untrained policy.")

        get_logger().info("Starting manual evaluation rollouts...")
        eval_env = self._env_creator({"base_env_config": self._config.eval_env, "worker_index": 999}) # Create single env instance
        n_runs = self._config.eval_env.n_eval_episodes
        all_rewards = []

        for episode in range(n_runs):
            obs, info = eval_env.reset()
            terminated = {"__all__": False}
            truncated = {"__all__": False}
            total_reward = 0.0
            step = 0
            # Store agent states if needed for policy mapping
            states = {agent_id: algo.get_policy(self.policy_mapping_fn(agent_id, None, None)).get_initial_state()
                      for agent_id in obs.keys()}
            rnn_used = bool(states[list(states.keys())[0]]) # Check if RNN state is present

            while not terminated["__all__"] and not truncated["__all__"]:
                actions = {}
                new_states = {}
                for agent_id, agent_obs in obs.items():
                     policy_id = self.policy_mapping_fn(agent_id, None, None) # Get policy for agent
                     if rnn_used:
                         actions[agent_id], states[agent_id], _ = algo.compute_single_action(
                             agent_obs, state=states[agent_id], policy_id=policy_id, explore=False
                         )
                     else:
                         actions[agent_id] = algo.compute_single_action(
                             agent_obs, policy_id=policy_id, explore=False
                         )

                obs, reward, terminated, truncated, info = eval_env.step(actions)
                # Aggregate reward if needed, PettingZoo usually returns dict
                total_reward += sum(reward.values()) # Example aggregation
                step += 1

                if record:
                    # RLlib doesn't have built-in VecMonitor style recording.
                    # You'd need to call env.render() and save frames manually.
                    # Example: frame = eval_env.render(); save_frame(frame)
                    pass # Add custom rendering/saving logic here

            all_rewards.append(total_reward)
            get_logger().info(f"Eval Episode {episode + 1}/{n_runs} finished. Reward: {total_reward}, Steps: {step}")

        eval_env.close()
        mean_reward = sum(all_rewards) / len(all_rewards) if all_rewards else 0
        get_logger().info(f"Mean evaluation reward over {n_runs} episodes: {mean_reward}")

        # Stop the algorithm
        algo.stop()

        fitness = mean_reward

        return fitness

        
    def _calc_seed(self, i: int) -> int:
        return self._config.seed + i

    # def _make_env(
    #     self,
    #     config: MjCambrianEnvConfig,
    #     n_envs: int,
    #     *,
    #     monitor: str | None = "monitor.csv",
    # ) -> VecEnv:
    #     assert n_envs > 0, f"n_envs must be > 0, got {n_envs}."

    #     # Create the environments
    #     envs = []
    #     for i in range(n_envs):
    #         wrappers = [w for w in self._config.trainer.wrappers.values() if w]
    #         wrapped_env = make_wrapped_env(
    #             config=config.copy(),
    #             name=self._config.expname,
    #             wrappers=wrappers,
    #             seed=self._calc_seed(i),
    #         )
    #         envs.append(wrapped_env)

    #     # Wrap the environments
    #     # Explicitly set start_method to spawn to avoid using forkserver on mac
    #     vec_env = (
    #         DummyVecEnv(envs)
    #         if n_envs == 1
    #         else SubprocVecEnv(envs, start_method="spawn")
    #     )
    #     if monitor is not None:
    #         vec_env = VecMonitor(vec_env, str(self._config.expdir / monitor))

    #     # Do an initial reset
    #     vec_env.reset()
    #     return vec_env

    # def _make_callback(self, env: VecEnv) -> CallbackList:
    #     """Makes the callbacks."""
    #     from functools import partial

    #     callbacks = []
    #     for callback in self._config.trainer.callbacks.values():
    #         # Only call the callback if it's a partial function
    #         if isinstance(callback, partial):
    #             callback = callback(env)
    #         callbacks.append(callback)

    #     return CallbackList(callbacks)

    def _make_model(self, env: VecEnv) -> MjCambrianModel:
        """This method creates the model."""
        return self._config.trainer.model(env=env)
    
    def _env_creator(
        self,
        env_config: MjCambrianEnvConfig,
        n_envs: int,
        *,
        monitor: str | None = "monitor.csv",
    ) -> gym.Env:
        """Creates, configures, and wraps a single environment instance for RLlib."""
        # env_config might contain worker_index, vector_index passed by RLlib
        seed = self._config.seed + env_config.get("worker_index", 0) # Example seeding

        # 1. Create the base environment instance
        # Use the appropriate config (train or eval)
        base_env_config = env_config.get("base_env_config", self._config.env)
        env = base_env_config.instance(base_env_config) # Assuming instance method exists

        for i in range(n_envs):
            wrappers = [w for w in self._config.trainer.wrappers.values() if w]
            wrapped_env = make_wrapped_env(
                config=env_config.copy(),
                name=self._config.expname,
                wrappers=wrappers,
                seed=self._calc_seed(i),
            )

        # 2. Apply necessary wrappers (ensure they are compatible with RLlib MARL)
        # Example: Assuming MjCambrianMultiAgentEnvWrapper makes it RLlib compatible
        # You might need different wrappers than SB3 used
        # env = MjCambrianMultiAgentEnvWrapper(env) # Adapt as needed
        # Add other wrappers like FrameStack, TorchToNumpy if required by the policy/RLlib
        # env = FrameStackObservation(env, stack_size=...) # Example
        # env = MjCambrianTorchToNumpyWrapper(env) # Example

        wrapped_env.reset(seed=seed) # Seed the single environment
        return wrapped_env

    def _register_env(self, name: str, base_env_config: MjCambrianEnvConfig):
         """Registers the environment creator function with RLlib."""
         register_env(name, lambda cfg: self._env_creator({**cfg, "base_env_config": base_env_config, "n_envs":1}))
    
     # --- Define Policy Mapping ---
    # Maps agent IDs to policy IDs
    def policy_mapping_fn(agent_id, episode, worker, **kwargs):
        if agent_id.startswith("agent_predator"):
            return "predator_policy"
        elif agent_id.startswith("agent_prey"):
            return "prey_policy"
        # Add mappings for goal, adversary if they are controlled agents
        # elif agent_id.startswith("goal"): return "static_policy" # Example
        else:
            raise ValueError(f"Unknown agent ID: {agent_id}")

    
    def _setup_rllib_algorithm(self, env_name: str):
        """Configures and builds the RLlib Algorithm."""

        # --- Define Policies ---
        # Assuming both predator and prey use the same policy architecture for now
        # Get observation and action spaces from a sample env instance
        # Note: Creating a dummy env here might be slow, consider alternatives
        # if performance is critical.
        sample_env = self._env_creator({"base_env_config": self._config.env})
        obs_space = sample_env.observation_space["agent_predator"] # Get space for one agent type
        act_space = sample_env.action_space["agent_predator"]      # Get space for one agent type
        sample_env.close() # Close the sample env if it holds resources

        policies = {
            "predator_policy": PolicySpec(observation_space=obs_space, action_space=act_space),
            "prey_policy": PolicySpec(observation_space=obs_space, action_space=act_space),
            # Add more policies if needed
        }

        # --- Configure Algorithm ---
        # Start with default config for the chosen algorithm
        if self._config.trainer.rllib_algorithm == "PPO":
            config = PPOConfig()
        # Add other algorithms like APEX_DDPG, IMPALA etc.
        # elif self._config.trainer.rllib_algorithm == "APEX_DDPG":
        #     config = ApexDDPGConfig()
        else:
            raise ValueError(f"Unsupported RLlib algorithm: {self._config.trainer.rllib_algorithm}")

        config = config.environment(env=env_name, disable_env_checking=True) # Use registered env
        config = config.framework("torch") # Or "tf"
        config = config.rollouts(
            num_rollout_workers=self._config.trainer.num_rollout_workers,
            # max_episode_steps or horizon might be set here or in env wrapper
            # rollout_fragment_length='auto' or specific value
        )
        config = config.multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapping_fn,
            # Optional: Specify which policies to train
            # policies_to_train=["predator_policy", "prey_policy"]
        )
        config = config.training(
             # gamma=0.99, lr=0.0001, etc. Set common training params here
             _enable_learner_api=False # Keep False for now unless using new API
        )
        config = config.resources(
            # num_gpus=1 # If using GPU
        )

        # Apply custom updates from config file
        config = config.from_dict(self._config.trainer.rllib_config_updates)

        # Build the Algorithm instance
        algo = config.build()
        return algo

    # --- Remove _make_model and _make_callback ---
    # def _make_model(...) -> MjCambrianModel: ...
    # def _make_callback(...) -> CallbackList: ...
