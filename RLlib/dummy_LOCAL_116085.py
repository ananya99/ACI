import numpy as np
import gym
import matplotlib.pyplot as plt
import os
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.ppo.torch.default_ppo_torch_rl_module import DefaultPPOTorchRLModule
from ray.rllib.algorithms.ppo.ppo_catalog import PPOCatalog
from ray.rllib.core.rl_module import RLModuleSpec
from ray.rllib.policy.policy import PolicySpec
from ray.tune.registry import register_env

# 1. Custom Catalog (unchanged)
class DictFlattenCatalog(PPOCatalog):
    def _determine_components_hook(self):
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

module_spec = RLModuleSpec(
    module_class=DefaultPPOTorchRLModule,
    catalog_class=DictFlattenCatalog,
)

# 2. Env registration (unchanged)
from mpe2 import simple_tag_v3
register_env("simple_tag", lambda cfg: PettingZooEnv(simple_tag_v3.env(
    num_good=cfg.get("num_good", 1),
    num_adversaries=cfg.get("num_adversaries", 1),
    num_obstacles=cfg.get("num_obstacles", 2),
    max_cycles=cfg.get("max_cycles", 25),
)))

# 3. Policy mapping (unchanged)
def policy_mapping_fn(agent_id, episode, **kwargs):
    return "adversary_policy" if agent_id.startswith("adversary") else "good_policy"

# 4. Build config (unchanged)
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
          policies_to_train=["good_policy"],
      )
)

# 5. Run training with logging and plotting
autooh = config.build()

# Prepare storage
agent_rewards = []
adv_rewards = []

# Open a CSV log file
log_path = "./RLlib/dummy_1.csv"
with open(log_path, "w") as log_file:
    # Header
    log_file.write("iteration,agent_0,adversary_0\n")

    print("Starting training loop…")
    for i in range(500):
        autooh.train()
        rewards = autooh.metrics.stats["env_runners"]["agent_episode_returns_mean"]
        ar = rewards.get('agent_0', np.nan)
        adr = rewards.get('adversary_0', np.nan)

        # Record
        agent_rewards.append(ar)
        adv_rewards.append(adr)

        # Log to file
        log_file.write(f"{i},{ar:.6f},{adr:.6f}\n")

        # Also print to console
        print(f"Iteration {i:03d}: agent reward={ar:.4f}; adv_reward={adr:.4f}")

# 6. Plot the learning curves
plt.figure(figsize=(8, 4))
plt.plot(agent_rewards, label="Agent 0 (prey)")
plt.plot(adv_rewards,  label="Adversary 0 (predator)")
plt.xlabel("Training Iteration")
plt.ylabel("Mean Episode Return")
plt.title("Learning Curves")
plt.legend()
plt.tight_layout()
plt.show()

fig_dir = "./RLlib/figures"
os.makedirs(fig_dir, exist_ok=True)

print(f"\nLogged results to {log_path}")