"""This module contains the trainer class for training and evaluating agents."""

from pathlib import Path
from typing import TYPE_CHECKING, Callable, Concatenate, Dict, Optional

from hydra_config import HydraContainerConfig, config_wrapper
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecEnv,
    VecMonitor,
)

from cambrian.envs.env import MjCambrianEnv, MjCambrianEnvConfig
from cambrian.ml.model import MjCambrianModel
from cambrian.utils import evaluate_policy
from cambrian.utils.logger import get_logger
from cambrian.utils.wrappers import make_wrapped_env
import os
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


class MjCambrianTrainer:
    """This is the trainer class for running training and evaluation.

    Args:
        config (MjCambrianConfig): The config to use for training and evaluation.
    """

    def __init__(self, config: "MjCambrianConfig"):
        self._config = config

        self._config.expdir.mkdir(parents=True, exist_ok=True)

        get_logger().info(f"Logging to {self._config.expdir / 'logs'}...")

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
        env = self._make_env(self._config.env, self._config.trainer.n_envs, monitor="monitor.csv", training_agent_name="agent_prey")
        env2 = self._make_env(self._config.env, self._config.trainer.n_envs, monitor="monitor2.csv", training_agent_name="agent_predator")
        envs = [env, env2]

        eval_env = self._make_env(self._config.eval_env, 1, monitor="eval_monitor.csv", training_agent_name="agent_prey")
        eval_env2 = self._make_env(self._config.eval_env, 1, monitor="eval2_monitor.csv", training_agent_name="agent_predator")
        eval_envs = [eval_env, eval_env2]

        callback = self._make_callback(eval_env)
        callback2 = self._make_callback(eval_env2)
        callbacks = [callback, callback2]

        # Save the eval environments xml
        for i, eval_env in enumerate(eval_envs):
            cambrian_eval_env: MjCambrianEnv = eval_env.envs[0].unwrapped
            cambrian_eval_env.xml.write(self._config.expdir / f"env_{i}.xml")
            with open(self._config.expdir / f"compiled_env_{i}.xml", "w") as f:
                f.write(cambrian_eval_env.spec.to_xml())

        agent_names = ['agent_prey', 'agent_predator']
        agent_models = {}
        for j, agent_name in enumerate(agent_names):
            model_path = os.path.join(self._config.expdir, agent_name+'_model.zip')
            agent_models[agent_name] = self._make_model(envs[j])
            agent_models[agent_name].save(model_path)
            print("created and saved initial model for agent:", agent_name, "at", model_path)

        iterations = 1
        total_timesteps = self._config.trainer.total_timesteps

        for i in range(iterations):
            for j in range(len(agent_names)):
                print("[INFO] Iteration: ", i)
                print("[INFO] Using model of agent:", agent_names[j], "while training agent:", agent_names[1-j])
                agent_models[agent_names[1-j]].learn(total_timesteps=total_timesteps, callback=callbacks[1-j])
                print("[INFO] Finished training the agent:", agent_names[1-j])
                print("[INFO] Saving model of",  agent_names[1-j], "to", self._config.expdir)
                agent_models[agent_names[1-j]].save(self._config.expdir, agent_names[1-j]+'_model.zip')

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

        eval_env = self._make_env(self._config.eval_env, 1, monitor="eval_monitor.csv", training_agent_name= 'agent_predator')
        cambrian_env: MjCambrianEnv = eval_env.envs[0].unwrapped
        model = self._make_model(eval_env)
        # if load_if_exists and (self._config.expdir / "best_model.zip").exists():
        get_logger().info("Loading best model...")
        model = model.load("/home/neo/Projects/vi/project/ACI/logs/2025-05-13/exp_detection/best_model.zip")

        # Save the eval environments xml
        cambrian_env: MjCambrianEnv = eval_env.envs[0].unwrapped
        cambrian_env.xml.write(self._config.expdir / "eval_env.xml")
        with open(self._config.expdir / "compiled_eval_env.xml", "w") as f:
            f.write(cambrian_env.spec.to_xml())

        n_runs = self._config.eval_env.n_eval_episodes
        filename = self._config.eval_env.save_filename
        record_kwargs = dict(
            path=self._config.expdir / filename,
            save_mode=self._config.eval_env.renderer.save_mode,
        )
        if not record:
            record_kwargs = None
        evaluate_policy(
            eval_env, model, n_runs, record_kwargs=record_kwargs, **callback_kwargs
        )

        # Calculate fitness
        fitness = self._config.trainer.fitness_fn(self._config)
        get_logger().info(f"Final Fitness: {fitness}")

        # Save the final fitness to a file
        with open(self._config.expdir / f"{filename}_fitness.txt", "w") as f:
            f.write(str(fitness))

        return fitness

    # ========

    def _calc_seed(self, i: int) -> int:
        return self._config.seed + i

    def _make_env(
        self,
        config: MjCambrianEnvConfig,
        n_envs: int,
        training_agent_name: str = None,
        *,
        monitor: str | None = "monitor.csv",
    ) -> VecEnv:
        assert n_envs > 0, f"n_envs must be > 0, got {n_envs}."

        # Create the environments
        envs = []
        for i in range(n_envs):
            wrappers = [w for w in self._config.trainer.wrappers.values() if w]
            wrapped_env = make_wrapped_env(
                config=config.copy(),
                name=self._config.expname,
                wrappers=wrappers,
                seed=self._calc_seed(i),
                training_agent_name = training_agent_name,
            )
            envs.append(wrapped_env)

        # Wrap the environments
        # Explicitly set start_method to spawn to avoid using forkserver on mac
        vec_env = (
            DummyVecEnv(envs)
            if n_envs == 1
            else SubprocVecEnv(envs, start_method="spawn")
        )
        if monitor is not None:
            vec_env = VecMonitor(vec_env, str(self._config.expdir / monitor))

        # Do an initial reset
        vec_env.reset()
        return vec_env

    def _make_callback(self, env: VecEnv) -> CallbackList:
        """Makes the callbacks."""
        from functools import partial

        callbacks = []
        for callback in self._config.trainer.callbacks.values():
            # Only call the callback if it's a partial function
            if isinstance(callback, partial):
                callback = callback(env)
            callbacks.append(callback)

        return CallbackList(callbacks)

    def _make_model(self, env: VecEnv) -> MjCambrianModel:
        """This method creates the model."""
        return self._config.trainer.model(env=env)
