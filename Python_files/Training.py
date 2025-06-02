# Standard library imports
import os
from typing import Dict, List

# Third-party imports
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import SubprocVecEnv # He is speeding up the training
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableMultiInputActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.monitor import Monitor  # Episode monitoring

# Local application/library specific imports
from Gym_Environment import HighSocietyEnv

# --- Constants ---
MODEL_CHECKPOINTS_DIR = "./model_checkpoints/"
MODEL_NAME_PREFIX = "hs_model"
FINAL_MODEL_SAVE_NAME = "high_society_trained_final"

# Hyperparameters
LEARNING_RATE = 1e-4
ENTHROPY = 0.1 # How much model explores (def = 0.01)
N_STEPS = 64 # After how many actions agent learns
BATCH_SIZE = 32 # Size of mini batch, so AI can quicker compute new nural network weights
N_EPOCHS = 4 # How many times it takes information from batches
FEATURES_DIM = 256  # Output dimension of the feature extractor
NET_ARCH_PI = [256, 256]  # Policy network architecture after feature extraction
NET_ARCH_VF = [256, 256]  # Value network architecture after feature extraction
TOTAL_TIMESTEPS = 200_000
CHECKPOINT_SAVE_FREQ = 500_000 # How often to save model



class CustomFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = FEATURES_DIM):
        super().__init__(observation_space, features_dim)
        
        # Update sizes to match actual tensor shapes
        self.obs_sizes = {
            "action_mask": 562,
            "num_players": 3,
            "current_bid": 1,
            "bidding_card": 14,
            "player_money": 1,
            "player_hand": 14,
            "player_score": 1,
            "highest_score": 1,  # Single value for highest score
            "poorest_money": 1,
            "score_of_winner": 1,
            "red_cards": 1
        }
        
        self.total_dim = sum(self.obs_sizes.values())

        self.shared_net = nn.Sequential(
            nn.Linear(self.total_dim, NET_ARCH_PI[0]),
            nn.ReLU(),
            nn.LayerNorm(NET_ARCH_PI[0]),
            nn.Linear(NET_ARCH_PI[0], features_dim),
            nn.ReLU()
        )

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Processes the dictionary observations and concatenates them.
        Ensures all tensors are 2D (batch_size, feature_size) before concatenation.
        """
        processed = []

        # Define the order of keys for processing and concatenation
        obs_order = [
            "action_mask", "num_players", "current_bid",
            "bidding_card", "player_money", "player_hand",
            "player_score", "highest_score", "poorest_money",
            "score_of_winner", "red_cards"
        ]

        for key in obs_order:
            tensor = observations[key]
            if isinstance(tensor, np.ndarray):
                tensor = torch.as_tensor(tensor, device=self.device)
            tensor = tensor.float()

            if key == "num_players":
                # Robust handling for one-hot or scalar
                if tensor.dim() == 3 and tensor.shape[1] == 1 and tensor.shape[2] == 3:
                    tensor = tensor.view(tensor.size(0), 3)
                elif tensor.dim() == 2 and tensor.shape[1] == 3:
                    pass
                else:
                    tensor = tensor.squeeze(-1)
                    tensor = F.one_hot(tensor.long(), num_classes=3).float()
                # shape: [batch_size, 3]
            elif key == "bidding_card":
                # Robust handling for one-hot or scalar
                if tensor.dim() == 3 and tensor.shape[1] == 1 and tensor.shape[2] == 14:
                    tensor = tensor.view(tensor.size(0), 14)
                elif tensor.dim() == 2 and tensor.shape[1] == 14:
                    pass
                else:
                    tensor = tensor.squeeze(-1)
                    tensor = F.one_hot(tensor.long(), num_classes=14).float()
                # shape: [batch_size, 14]
            elif key in ["action_mask", "player_hand"]:
                if tensor.dim() == 1:
                    tensor = tensor.unsqueeze(1)
            else:
                if tensor.dim() == 1:
                    tensor = tensor.unsqueeze(1)
                elif tensor.dim() > 2:
                    tensor = tensor.view(tensor.size(0), -1)

            processed.append(tensor)

        processed = [t if t.dim() == 2 else t.view(t.size(0), -1) for t in processed]
        concatenated = torch.cat(processed, dim=1)
        assert concatenated.size(1) == self.total_dim, f"Concatenated dimension mismatch. Expected {self.total_dim}, got {concatenated.size(1)}"
        return self.shared_net(concatenated)


def make_env(env_id, rank, seed=0): #Utility function for multiprocessed env.
    def _init():
        env = HighSocietyEnv(num_players=random.randint(3, 5))

        # Wrap env with ActionMasker to enforce legal actions
        env = ActionMasker(env, lambda e: e.unwrapped._create_action_mask())

        env.reset(seed=seed + rank)
        return env
    return _init

def train_agent():
    n_envs = 4 # For Standard_F4s_v2 (my CPU)

    print(f"Creating {n_envs} parallel environments...")
    # Create and monitor each sub-environment
    env_fns = [make_env("HighSocietyEnv", i, seed=0) for i in range(n_envs)]
    monitored_envs = [lambda fn=fn: Monitor(fn()) for fn in env_fns]
    env = SubprocVecEnv(monitored_envs)

    print("Setting up policy and model...")

    policy_kwargs = dict(
        features_extractor_class=CustomFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=FEATURES_DIM),
        net_arch=dict(pi=NET_ARCH_PI, vf=NET_ARCH_VF)
    )

    model = MaskablePPO(
        MaskableMultiInputActorCriticPolicy,
        env,
        verbose=1,
        tensorboard_log="./logs/hs_ppo_run",
        policy_kwargs=policy_kwargs,
        ent_coef=ENTHROPY,
        learning_rate=LEARNING_RATE,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        device="cpu"
    )

    print("Setting up callbacks...")
    # Checkpoint callback to save model periodically
    # Ensure the save directory exists
    os.makedirs(MODEL_CHECKPOINTS_DIR, exist_ok=True)
    checkpoint_callback = CheckpointCallback(
        save_freq=CHECKPOINT_SAVE_FREQ,
        save_path=MODEL_CHECKPOINTS_DIR,
        name_prefix=MODEL_NAME_PREFIX,
        save_replay_buffer=False, # Typically not needed for PPO, saves space
        save_vecnormalize=False   # If VecNormalize is used, set to True
    )

    print(f"Starting training for {TOTAL_TIMESTEPS} timesteps...")
    # Train the agent
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=checkpoint_callback,
        progress_bar=True,
        reset_num_timesteps=True # Start timesteps count from 0 for this learn call
    )

    print("Training complete.")
    # Save the final trained model
    final_model_path = os.path.join(MODEL_CHECKPOINTS_DIR, FINAL_MODEL_SAVE_NAME)
    model.save(final_model_path)
    print(f"Final model saved to {final_model_path}")


if __name__ == "__main__":
    # Entry point of the script
    try:
        train_agent()
    except Exception as e:
        print(f"An error occurred during training: {e}")
        # In a production setting, you might use a more sophisticated logging framework
        # or error reporting mechanism here.
        raise # Re-raise the exception to see the full traceback for debugging
