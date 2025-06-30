#!/usr/bin/env python3
"""
Test Trained Traffic Signal RL Agent

This script loads your trained AI agent and demonstrates it controlling
traffic lights in real-time. You can watch your AI make intelligent
decisions to minimize traffic congestion!
"""

import os
import sys
import time
import numpy as np

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from stable_baselines3 import PPO
from environments.sumo_traffic_env import SumoTrafficEnv


def load_trained_agent(model_path):
    """Load the trained PPO agent."""
    print(f"🤖 Loading trained agent from: {model_path}")
    model = PPO.load(model_path)
    print("✅ AI agent loaded successfully!")
    return model


def create_demo_environment():
    """Create environment for demonstration with longer episodes."""
    env = SumoTrafficEnv(
        sumo_config_file="sumo_scenarios/cross_intersection.sumocfg",
        step_duration=5,  # 5 seconds per RL step
        episode_length=600  # 10 minutes of simulation time for good demo
    )
    return env


def demonstrate_ai_control(model, env, episodes=3):
    """
    Demonstrate the AI controlling traffic for multiple episodes.
    
    Args:
        model: Trained PPO agent
        env: SUMO environment
        episodes: Number of episodes to run
    """
    print(f"\n🚦 Starting AI Traffic Control Demo")
    print(f"Running {episodes} episodes (10 minutes of traffic each)")
    print("=" * 60)
    
    episode_rewards = []
    episode_stats = []
    
    for episode in range(episodes):
        print(f"\n📋 Episode {episode + 1}/{episodes}")
        print("-" * 40)
        
        # Reset environment
        obs, info = env.reset()
        total_reward = 0
        step_count = 0
        
        # Track statistics
        total_waiting_time = 0
        total_vehicles_served = 0
        phase_changes = 0
        last_phase = None
        
        done = False
        while not done:
            # AI makes decision
            action, _states = model.predict(obs, deterministic=True)
            
            # Execute action
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Update statistics
            total_reward += reward
            step_count += 1
            
            if 'total_waiting_time' in info:
                total_waiting_time = info['total_waiting_time']
            
            if 'current_phase' in info:
                current_phase = info['current_phase']
                if last_phase is not None and current_phase != last_phase:
                    phase_changes += 1
                last_phase = current_phase
            
            # Print periodic updates
            if step_count % 20 == 0:  # Every 100 seconds of simulation
                print(f"  Step {step_count:3d}: Reward = {total_reward:6.1f}, "
                      f"Waiting time = {total_waiting_time:5.1f}s, "
                      f"Phase = {info.get('current_phase', 'N/A')}")
        
        # Episode summary
        avg_reward_per_step = total_reward / step_count if step_count > 0 else 0
        episode_rewards.append(total_reward)
        
        episode_stats.append({
            'episode': episode + 1,
            'total_reward': total_reward,
            'avg_reward_per_step': avg_reward_per_step,
            'total_waiting_time': total_waiting_time,
            'phase_changes': phase_changes,
            'steps': step_count
        })
        
        print(f"\n📊 Episode {episode + 1} Results:")
        print(f"   Total reward: {total_reward:.1f}")
        print(f"   Average reward per step: {avg_reward_per_step:.2f}")
        print(f"   Final waiting time: {total_waiting_time:.1f} seconds")
        print(f"   Phase changes: {phase_changes}")
        print(f"   Simulation steps: {step_count}")
    
    # Overall summary
    print("\n" + "=" * 60)
    print("🎯 OVERALL AI PERFORMANCE SUMMARY")
    print("=" * 60)
    
    avg_total_reward = np.mean(episode_rewards)
    std_total_reward = np.std(episode_rewards)
    
    print(f"📈 Average episode reward: {avg_total_reward:.1f} ± {std_total_reward:.1f}")
    print(f"🎯 Best episode reward: {max(episode_rewards):.1f}")
    print(f"📉 Worst episode reward: {min(episode_rewards):.1f}")
    
    avg_waiting = np.mean([s['total_waiting_time'] for s in episode_stats])
    avg_phase_changes = np.mean([s['phase_changes'] for s in episode_stats])
    
    print(f"⏰ Average waiting time per episode: {avg_waiting:.1f} seconds")
    print(f"🔄 Average phase changes per episode: {avg_phase_changes:.1f}")
    
    print("\n💡 What this means:")
    print(f"   • Your AI minimizes vehicle waiting (negative reward closer to 0 is better)")
    print(f"   • Lower waiting times = more efficient traffic flow")
    print(f"   • Fewer phase changes = smoother operation")
    
    return episode_stats


def test_ai_intelligence(model, env):
    """
    Test specific scenarios to show AI intelligence.
    """
    print("\n🧠 TESTING AI INTELLIGENCE")
    print("=" * 40)
    print("Let's see how your AI handles specific traffic scenarios...")
    
    # Reset environment
    obs, info = env.reset()
    
    # Run a few steps and analyze decisions
    for step in range(10):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            break
            
        # Analyze current state
        incoming_vehicles = obs[:4]  # N, E, S, W incoming
        waiting_times = obs[8:12]   # N, E, S, W waiting times
        current_phase = np.argmax(obs[16:20])  # Current traffic light phase
        
        print(f"\nStep {step + 1}:")
        print(f"  🚗 Vehicles waiting: N={incoming_vehicles[0]:.0f}, E={incoming_vehicles[1]:.0f}, "
              f"S={incoming_vehicles[2]:.0f}, W={incoming_vehicles[3]:.0f}")
        print(f"  ⏰ Waiting times: N={waiting_times[0]:.1f}s, E={waiting_times[1]:.1f}s, "
              f"S={waiting_times[2]:.1f}s, W={waiting_times[3]:.1f}s")
        print(f"  🚦 AI chose action {action} (phase {current_phase})")
        print(f"  🎁 Reward: {reward:.1f}")
        
        # Analyze AI's decision
        max_waiting_direction = np.argmax(waiting_times)
        direction_names = ['North', 'East', 'South', 'West']
        
        if incoming_vehicles[max_waiting_direction] > 0:
            print(f"  🤖 AI Logic: Most congestion in {direction_names[max_waiting_direction]} direction")


def main():
    """Main demonstration function."""
    print("🚦 Trained AI Traffic Control Demonstration")
    print("This will show your AI controlling traffic lights intelligently!")
    print()
    
    # Find the trained model
    model_path = "models/ppo_traffic_final_20250630_183234.zip"
    
    if not os.path.exists(model_path):
        # Try to find the latest checkpoint
        checkpoint_files = [f for f in os.listdir("models/checkpoints/") if f.endswith('.zip')]
        if checkpoint_files:
            latest_checkpoint = sorted(checkpoint_files)[-1]
            model_path = f"models/checkpoints/{latest_checkpoint}"
            print(f"⚠️  Using latest checkpoint: {model_path}")
        else:
            print("❌ No trained model found!")
            return
    
    try:
        # Load trained agent
        model = load_trained_agent(model_path)
        
        # Create environment
        print("\n🏗️  Creating demonstration environment...")
        env = create_demo_environment()
        print("✅ Environment ready!")
        
        # Run demonstration
        episode_stats = demonstrate_ai_control(model, env, episodes=3)
        
        # Test AI intelligence
        test_ai_intelligence(model, env)
        
        print("\n🎉 Demonstration completed!")
        print("\nYour AI has successfully learned to control traffic signals!")
        print("Next steps:")
        print("  1. Compare against baseline controllers")
        print("  2. Create visualizations")
        print("  3. Build evaluation metrics")
        
    except Exception as e:
        print(f"❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        print("\n🧹 Cleaning up...")
        try:
            env.close()
        except:
            pass
        print("✅ Demo completed!")


if __name__ == "__main__":
    main() 