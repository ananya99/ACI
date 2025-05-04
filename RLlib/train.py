import numpy as np
import gym
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.ppo.torch.default_ppo_torch_rl_module import DefaultPPOTorchRLModule
from ray.rllib.algorithms.ppo.ppo_catalog import PPOCatalog
from ray.rllib.core.rl_module import RLModuleSpec
from ray.rllib.policy.policy import PolicySpec
from ray.tune.registry import register_env

# 1. Create a custom Catalog to flatten Dict observations
class DictFlattenCatalog(PPOCatalog):
    def _determine_components_hook(self):
        """
        If the observation space is a gym.spaces.Dict, flatten all sub-spaces into
        a single vector input for the encoder network. Otherwise, fall back to default.
        """
        obs_space = self.observation_space
        if isinstance(obs_space, gym.spaces.Dict):
            total_dim = sum(int(np.product(sub.shape)) for sub in obs_space.spaces.values())
            self._encoder_config = {
                "fcnet_hiddens": [256, 256],
                "fcnet_activation": "relu",
                "input_dims": total_dim,
            }
        else:
            super()._determine_components_hook()

# 2. Define an RLModuleSpec using our custom catalog
module_spec = RLModuleSpec(
    module_class=DefaultPPOTorchRLModule,
    catalog_class=DictFlattenCatalog,
)

# 3. Environment registration
from mpe2 import simple_tag_v3
register_env("simple_tag", lambda cfg: PettingZooEnv(simple_tag_v3.env(
    num_good=cfg.get("num_good", 1),
    num_adversaries=cfg.get("num_adversaries", 3),
    num_obstacles=cfg.get("num_obstacles", 2),
    max_cycles=cfg.get("max_cycles", 25),
)))

# 4. Policy mapping function with the new signature
def policy_mapping_fn(agent_id, episode, **kwargs):
    return "adversary_policy" if agent_id.startswith("adversary") else "good_policy"

# 5. Build the PPOConfig with env runners and RLModule
config = (
    PPOConfig()
      .environment("simple_tag", env_config={
          "num_good": 1,
          "num_adversaries": 1,
          "num_obstacles": 2,
          "max_cycles": 25,
      })
      .framework("torch")
      .api_stack(
          enable_rl_module_and_learner=True,
          enable_env_runner_and_connector_v2=True,
      )
      .env_runners(num_env_runners=2, num_envs_per_env_runner=2)
      .rl_module(rl_module_spec=module_spec)
      .resources(num_gpus=1)
      .training(train_batch_size=4000)
      .multi_agent(
          policies={
              "good_policy": PolicySpec(None, None, None, {}),
              "adversary_policy": PolicySpec(None, None, None, {}),
          },
          policy_mapping_fn=policy_mapping_fn,
      )
)

# 6. Run training
autooh = config.build()
print("Starting training loop…")
for i in range(200):
    autooh.train()
    rewards = autooh.metrics.stats["env_runners"]["agent_episode_returns_mean"]
    agent_reward = rewards['agent_0']
    adv_reward = rewards['adversary_0']
    print(f"Iteration {i:03d}: agent reward={agent_reward}; adv_reward={adv_reward}")
    