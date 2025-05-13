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

# Local application/library specific imports
from Gym_Environment import HighSocietyEnv

# --- Constants ---
MODEL_CHECKPOINTS_DIR = "./model_checkpoints/"
MODEL_NAME_PREFIX = "hs_model"
FINAL_MODEL_SAVE_NAME = "high_society_trained_final"

# Hyperparameters
LEARNING_RATE = 3e-4
ENTHROPY = 0.05 # How much model explores (def = 0.01)
N_STEPS = 64 # After how many actions agent learns
BATCH_SIZE = 32 # Size of mini batch, so AI can quicker compute new nural network weights
N_EPOCHS = 4 # How many times it takes information from batches
FEATURES_DIM = 256  # Output dimension of the feature extractor
NET_ARCH_PI = [256, 256]  # Policy network architecture after feature extraction
NET_ARCH_VF = [256, 256]  # Value network architecture after feature extraction
TOTAL_TIMESTEPS = 100_000
CHECKPOINT_SAVE_FREQ = 500_000 # How often to save model



class CustomFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = FEATURES_DIM):
        super().__init__(observation_space, features_dim)
        
        # Update sizes to match actual tensor shapes
        self.obs_sizes = {
            "action_mask": 562,  # Matches actual shape from error
            "num_players": 16,   # From one-hot encoding (4x4)
            "current_bid": 1,    # Single value
            "bidding_card": 196, # From actual shape
            "player_money": 1,   # Single value
            "player_hand": 14,   # Binary vector
            "player_score": 1,   # Single value
            "other_players": 8,  # 4 players × 2 values
            "deck_remaining": 1  # Single value
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
        processed = []
        
        for key in [
            "action_mask", "num_players", "current_bid",
            "bidding_card", "player_money", "player_hand",
            "player_score", "other_players", "deck_remaining"
        ]:
            tensor = observations[key]
            if isinstance(tensor, np.ndarray):
                tensor = torch.as_tensor(tensor, device=self.device)
                
            # Handle different space types
            if key == "num_players":
                tensor = F.one_hot(tensor.long(), num_classes=4).float()
            elif key == "bidding_card":
                tensor = F.one_hot(tensor.long(), num_classes=14).float()
            
            # Flatten and ensure 2D
            if tensor.dim() > 2:
                tensor = tensor.reshape(tensor.size(0), -1)
            elif tensor.dim() == 1:
                tensor = tensor.unsqueeze(0)
                
            processed.append(tensor)
            
        # Concatenate all processed tensors
        concatenated = torch.cat(processed, dim=1)
        
        # Ensure correct shape for network
        if concatenated.size(1) != self.total_dim:
            raise ValueError(f"Input dimension mismatch. Expected {self.total_dim}, got {concatenated.size(1)}")
        
        return self.shared_net(concatenated)


def make_env(env_id, rank, seed=0): #Utility function for multiprocessed env.
    def _init():
        # Create your environment instance inside this function
        # Pass seed + rank to the environment reset for reproducibility
        env = HighSocietyEnv(num_players=random.randint(3, 5))

        # Apply wrappers *inside* _init()
        def mask_fn(env):
             return env.unwrapped._create_action_mask()
        env = ActionMasker(env, mask_fn)

        # Call reset here with seed
        env.reset(seed=seed + rank) # Use a unique seed for each environment instance

        return env
    return _init

def train_agent():
    n_envs = 4 # For Standard_F4s_v2 (my CPU)

    print(f"Creating {n_envs} parallel environments...")
    env = SubprocVecEnv([make_env("HighSocietyEnv", i, seed=0) for i in range(n_envs)])

    print("Setting up policy and model...")



    policy_kwargs = dict(
        features_extractor_class=CustomFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=FEATURES_DIM),
        net_arch=dict(pi=NET_ARCH_PI, vf=NET_ARCH_VF)
    )

    # Initialize the MaskablePPO model
    # MaskablePPO is used for environments with action masking.
    model = MaskablePPO(
        MaskableMultiInputActorCriticPolicy,  # Policy designed for multi-input and masking
        env,
        verbose=0,  # 0 for no output, 1 for info, 2 for debug
        tensorboard_log=None,  # Disable TensorBoard logging
        policy_kwargs=policy_kwargs,
        ent_coef=ENTHROPY,
        learning_rate=LEARNING_RATE,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        device="cpu" # Force CPU usage
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
