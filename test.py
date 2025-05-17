from pettingzoo.mpe import simple_tag_v3
from pettingzoo.utils.conversions import aec_to_parallel
from pettingzoo.utils.wrappers import BaseParallelWrapper
import supersuit as ss

import numpy as np
import gymnasium as gym
from gymnasium import spaces

import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.distributions import Distribution

# --- Observation padding wrapper ---
class StackObservationsWrapper(BaseParallelWrapper):
    def __init__(self, env):
        super().__init__(env)
        sample_agent = env.possible_agents[0]
        self.obs_dim = env.observation_space(sample_agent).shape[0]
        self.unified_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_dim,),
            dtype=np.float32
        )

    def observation_space(self, agent):
        return self.unified_space

    def reset(self, seed=None, options=None):
        return self.env.reset(seed=seed, options=options)

    def step(self, actions):
        return self.env.step(actions)

# --- Custom policy network ---
class SwitchingMLP(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
        )
        self.predator_actor = nn.Linear(hidden_dim, action_dim)
        self.prey_actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        )
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, obs):
        features = self.shared(obs)
        pred_mask = ~torch.all(obs[:, :obs.shape[1]//2] == 0, dim=1)
        
        # Initialize with proper batch dimensions
        actor_output = torch.zeros((obs.size(0), self.predator_actor.out_features), 
                            device=obs.device)
        
        if pred_mask.any():
            actor_output[pred_mask] = self.predator_actor(features[pred_mask])
        if (~pred_mask).any():
            actor_output[~pred_mask] = self.prey_actor(features[~pred_mask])
        
        values = self.critic(features).squeeze(-1)
        return actor_output, values

# --- Custom policy for SB3 ---
class AECSwitchingPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, net_arch=[], activation_fn=nn.ReLU)
        obs_dim = self.features_extractor.features_dim
        act_dim = self.action_net.out_features
        self.switching_mlp = SwitchingMLP(obs_dim, act_dim)
        
        self.action_net = nn.Identity()
        self.value_net = nn.Identity()

    def _get_action_dist_from_latent(self, latent_pi: torch.Tensor) -> Distribution:
        return self.action_dist.proba_distribution(action_logits=latent_pi)

    def forward(self, obs, deterministic=False):
        features = self.extract_features(obs)
        latent_pi, values = self.switching_mlp(features)
        dist = self._get_action_dist_from_latent(latent_pi)
        actions = dist.get_actions(deterministic=deterministic)
        log_prob = dist.log_prob(actions)
        return actions, values, log_prob

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        features = self.extract_features(obs)
        latent_pi, values = self.switching_mlp(features)
        dist = self._get_action_dist_from_latent(latent_pi)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        return values, log_prob, entropy

    def predict_values(self, obs: torch.Tensor) -> torch.Tensor:
        features = self.extract_features(obs)
        _, values = self.switching_mlp(features)
        return values

# --- Environment setup ---
def make_wrapped_simple_tag():
    aec_env = simple_tag_v3.env(
        num_good=1, num_adversaries=1, num_obstacles=0,
        max_cycles=25, continuous_actions=False
    )
    
    parallel_env = aec_to_parallel(aec_env)
    padded_env = ss.pad_observations_v0(parallel_env)
    stacked_env = StackObservationsWrapper(padded_env)
    vec_env = ss.pettingzoo_env_to_vec_env_v1(stacked_env)
    sb3_env = ss.concat_vec_envs_v1(vec_env, num_vec_envs=1, num_cpus=1, base_class='stable_baselines3')
    return sb3_env

# --- Training ---
env = make_wrapped_simple_tag()

model = PPO(
    policy=AECSwitchingPolicy,
    env=env,
    verbose=1,
    batch_size=256,
    n_steps=2048,
    n_epochs=10,
    tensorboard_log="./tag_tensorboard"
)

model.learn(total_timesteps=1000)

def calculate_success_rate(model, num_episodes=100):
    # Create a parallel (not vectorized) environment for evaluation
    aec_env = simple_tag_v3.env(
        num_good=1, num_adversaries=1, num_obstacles=0,
        max_cycles=25, continuous_actions=False
    )
    parallel_env = aec_to_parallel(aec_env)
    padded_env = ss.pad_observations_v0(parallel_env)
    env = StackObservationsWrapper(padded_env)
    
    successes = 0
    
    for i in range(num_episodes):
        obs, _ = env.reset()
        prey_alive = True
        print(f'Episode {i+1}')
        
        while prey_alive:
            actions = {}
            # Get actions for all predators
            for agent in env.agents:
                if "adversary" in agent:
                    agent_obs = obs[agent]
                    # Add batch dimension and convert to tensor
                    print(f"Agent {agent} observation: {agent_obs}")
                    tensor_obs = torch.tensor(agent_obs).float().unsqueeze(0)
                    action = model.predict(tensor_obs, deterministic=True)[0]
                    actions[agent] = action.item()
                else:
                    # Let prey take random actions
                    actions[agent] = env.action_space(agent).sample().item()
            
            print(f"Actions: {actions}")
            obs, _, terminations, _, _ = env.step(actions)
            
            if terminations.get("agent_0", False):
                successes += 1
                prey_alive = False
                
    return successes / num_episodes

print(f"Prey capture rate: {calculate_success_rate(model):.2%}")