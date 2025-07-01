#!/usr/bin/env python3
"""
Training Script for Agent

This script creates and trains a Proximal Policy Optimization (PPO) agent
to control traffic lights at our cross intersection. The agent learns by
interacting with the SUMO simulation thousands of times, getting better
at minimizing traffic congestion through experience.

"""

import os
import sys
import yaml
from datetime import datetime
import numpy as np

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import torch

from environments.sumo_traffic_env import SumoTrafficEnv


def load_config(config_path="configs/training_config.yaml"):
    """Load training configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_environment(config):
    """
    Create and wrap the SUMO traffic environment.
    
    The Monitor wrapper tracks episode statistics (rewards, lengths)
    The DummyVecEnv wrapper makes it compatible with Stable Baselines3
    """
    # Create the base environment
    env = SumoTrafficEnv(
        sumo_config_file="sumo_scenarios/cross_intersection.sumocfg",
        step_duration=config['environment']['step_size'],
        episode_length=config['environment']['episode_length']
    )
    
    # Wrap with Monitor to track episode statistics
    env = Monitor(env)
    
    # Wrap with DummyVecEnv (required by Stable Baselines3)
    env = DummyVecEnv([lambda: env])
    
    return env


def create_agent(env, config):
    """
    Create the PPO agent.
    
    PPO is a policy gradient method that's stable and sample efficient.
    It learns a policy (what action to take in each state) and a value function
    (how good each state is) simultaneously. It's gradient based, so it can learn
    from the mistakes of the agent. Good balance between exploration and exploitation.
    """
    # Extract algorithm configuration
    algo_config = config['algorithm']
    network_config = config['network']
    
    # Create PPO agent
    model = PPO(
        policy="MlpPolicy",  # Multi-Layer Perceptron policy network
        env=env,
        learning_rate=algo_config['learning_rate'],
        batch_size=algo_config['batch_size'],
        n_epochs=algo_config['n_epochs'],
        verbose=1,  # Print training progress
        tensorboard_log="./logs/tensorboard/",  # For monitoring training
        device="auto"  # Use GPU if available, otherwise CPU
    )
    
    return model


def setup_callbacks(config):
    """
    Setup training callbacks for monitoring and checkpointing.
    
    Callbacks are functions that run during training to:
    - Save model checkpoints periodically
    - Evaluate performance during training
    - Log metrics for analysis
    """
    # Create directories if they don't exist
    os.makedirs("models/checkpoints", exist_ok=True)
    os.makedirs("logs/eval", exist_ok=True)
    
    # Checkpoint callback - saves model every N steps
    checkpoint_callback = CheckpointCallback(
        save_freq=config['training']['save_frequency'],
        save_path="models/checkpoints/",
        name_prefix="ppo_traffic_model"
    )
    
    return [checkpoint_callback]


def train_agent():
    """Main training function."""
    print("🚀 Starting Traffic Signal RL Training")
    print("=" * 50)
    
    # Load configuration
    print("📋 Loading configuration...")
    config = load_config()
    print(f"✅ Config loaded: {config['algorithm']['name']} for {config['algorithm']['total_timesteps']} timesteps")
    
    # Create environment
    print("\nCreating environment...")
    env = create_environment(config)
    print("✅ Environment created and wrapped")
    
    # Create agent
    print("\nCreating RL agent...")
    model = create_agent(env, config)
    print(f"✅ {config['algorithm']['name']} agent created")
    print(f"   Learning rate: {config['algorithm']['learning_rate']}")
    print(f"   Batch size: {config['algorithm']['batch_size']}")
    
    # Setup callbacks
    print("\nSetting up training callbacks...")
    callbacks = setup_callbacks(config)
    print("✅ Callbacks configured")
    
    # Start training
    print("\nStarting training...")
    print(f"Total timesteps: {config['algorithm']['total_timesteps']}")
    
    start_time = datetime.now()
    
    # THE MAIN TRAINING LOOP
    model.learn(
        total_timesteps=config['algorithm']['total_timesteps'],
        callback=callbacks,
        progress_bar=False  # Disable progress bar to avoid display issues
    )
    
    end_time = datetime.now()
    training_duration = end_time - start_time
    
    # Save final model
    model_filename = f"models/ppo_traffic_final_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
    model.save(model_filename)
    
    print("\nTraining completed!")
    print(f"Training duration: {training_duration}")
    print(f"Final model saved as: {model_filename}")
    print(f"Tensorboard logs: ./logs/tensorboard/")
    print(f"Checkpoints saved in: models/checkpoints/")
    
    # Close environment
    env.close()
    
    return model, model_filename


if __name__ == "__main__":
    print("Training")
    print()
    
    # Check if SUMO is available
    try:
        import traci
        print("✅ SUMO/TraCI available")
    except ImportError:
        print("❌ SUMO/TraCI not found. Please install SUMO first.")
        sys.exit(1)
    
    # Check if PyTorch is using GPU
    if torch.cuda.is_available():
        print(f"GPU available: {torch.cuda.get_device_name()}")
    else:
        print("Using CPU for training")
    
    print()
    
    # Run training
    try:
        model, model_path = train_agent()
        print("\nTraining completed successfully!")
        print(f"Trained agent is ready at: {model_path}")
        
    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc() 