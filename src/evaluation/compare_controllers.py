"""
Controller Comparison Script

This script compares the performance of different traffic control strategies:
1. Trained RL Agent
2. Fixed-Time Controller
3. Adaptive Fixed-Time Controller

It runs them under identical traffic conditions and collects metrics to prove
which approach works better.
"""

import sys
import os
import numpy as np
import pandas as pd
import time
from typing import Dict, List, Tuple
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from environments.sumo_traffic_env import SumoTrafficEnv
from agents.fixed_time_controller import FixedTimeController, AdaptiveFixedTimeController, ActuatedController

try:
    from stable_baselines3 import PPO
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    print("Warning: Stable Baselines3 not available. Cannot test RL agent.")


class TrafficControllerComparison:
    """
    Comprehensive comparison of different traffic control strategies.
    """
    
    def __init__(self, sumo_config_file: str = "sumo_scenarios/cross_intersection.sumocfg"):
        self.sumo_config_file = sumo_config_file
        self.results = {}
        
    def test_controller(self, controller, controller_name: str, num_episodes: int = 5, 
                       episode_length: int = 120, verbose: bool = True) -> Dict:
        """
        Test a single controller and collect performance metrics.
        
        Args:
            controller: The controller to test (RL agent or baseline)
            controller_name: Name for reporting
            num_episodes: Number of test episodes
            episode_length: Length of each episode in steps
            verbose: Whether to print progress
            
        Returns:
            Dictionary of performance metrics
        """
        if verbose:
            print(f"\nTesting {controller_name}")
            print("=" * 50)
        
        # Create environment
        env = SumoTrafficEnv(
            sumo_config_file=self.sumo_config_file,
            step_duration=5,
            episode_length=episode_length
        )
        
        episode_rewards = []
        episode_waiting_times = []
        episode_phase_changes = []
        episode_throughput = []
        
        try:
            for episode in range(num_episodes):
                if verbose:
                    print(f"  Episode {episode + 1}/{num_episodes}")
                
                # Reset environment and controller
                obs, _ = env.reset()
                if hasattr(controller, 'reset'):
                    controller.reset()
                
                episode_reward = 0
                total_waiting_time = 0
                phase_changes = 0
                last_action = None
                vehicles_passed = 0
                
                for step in range(episode_length):
                    # Get action from controller
                    if hasattr(controller, 'predict'):
                        # RL agent
                        action, _ = controller.predict(obs, deterministic=True)
                    else:
                        # Baseline controller
                        action = controller.get_action(obs)
                    
                    # Track phase changes
                    if last_action is not None and action != last_action:
                        phase_changes += 1
                    last_action = action
                    
                    # Step environment
                    obs, reward, terminated, truncated, info = env.step(action)
                    episode_reward += reward
                    
                    # Extract waiting time from observation
                    current_waiting = np.sum(obs[8:16])  # Waiting times from state
                    total_waiting_time += current_waiting
                    
                    # Count vehicles that passed (simplified metric)
                    vehicles_passed += info.get('vehicles_passed', 0)
                    
                    if terminated or truncated:
                        break
                
                # Store episode results
                episode_rewards.append(episode_reward)
                episode_waiting_times.append(total_waiting_time / episode_length)
                episode_phase_changes.append(phase_changes)
                episode_throughput.append(vehicles_passed)
                
                if verbose:
                    print(f"    Reward: {episode_reward:.1f}, "
                          f"Avg Waiting: {total_waiting_time/episode_length:.1f}s, "
                          f"Phase Changes: {phase_changes}")
        
        finally:
            env.close()
        
        # Calculate summary statistics
        results = {
            'controller_name': controller_name,
            'num_episodes': num_episodes,
            'episode_length': episode_length,
            'avg_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'avg_waiting_time': np.mean(episode_waiting_times),
            'std_waiting_time': np.std(episode_waiting_times),
            'avg_phase_changes': np.mean(episode_phase_changes),
            'std_phase_changes': np.std(episode_phase_changes),
            'avg_throughput': np.mean(episode_throughput),
            'std_throughput': np.std(episode_throughput),
            'episode_rewards': episode_rewards,
            'episode_waiting_times': episode_waiting_times,
            'episode_phase_changes': episode_phase_changes,
            'episode_throughput': episode_throughput
        }
        
        if verbose:
            print(f"  Results Summary:")
            print(f"    Average Reward: {results['avg_reward']:.2f} ± {results['std_reward']:.2f}")
            print(f"    Average Waiting Time: {results['avg_waiting_time']:.2f} ± {results['std_waiting_time']:.2f}s")
            print(f"    Average Phase Changes: {results['avg_phase_changes']:.1f} ± {results['std_phase_changes']:.1f}")
            print(f"    Average Throughput: {results['avg_throughput']:.1f} ± {results['std_throughput']:.1f}")
        
        return results
    
    def run_comparison(self, rl_model_path: str = None, num_episodes: int = 5, 
                      episode_length: int = 120) -> Dict:
        """
        Run complete comparison of all available controllers.
        
        Args:
            rl_model_path: Path to trained RL model
            num_episodes: Number of episodes per controller
            episode_length: Steps per episode
            
        Returns:
            Complete comparison results
        """
        print("TRAFFIC CONTROLLER COMPARISON")
        print("=" * 60)
        print(f"Testing conditions:")
        print(f"  Episodes per controller: {num_episodes}")
        print(f"  Episode length: {episode_length} steps ({episode_length * 5} seconds)")
        print(f"  SUMO config: {self.sumo_config_file}")
        
        controllers_to_test = []
        
        # 1. Fixed-Time Controller (Traditional)
        fixed_controller = FixedTimeController(
            phase_durations=[6, 6, 6, 6],  # 30 seconds per phase
            step_duration=5
        )
        controllers_to_test.append((fixed_controller, "Fixed-Time (30s cycle)"))
        
        # 2. Adaptive Fixed-Time Controller
        adaptive_controller = AdaptiveFixedTimeController(
            north_south_duration=8,  # 40 seconds for main road
            east_west_duration=4,    # 20 seconds for side road
            step_duration=5
        )
        controllers_to_test.append((adaptive_controller, "Adaptive Fixed-Time"))
        
        # 3. Actuated Controller (Industry Standard)
        actuated_controller = ActuatedController(
            min_green_time=2,    # 10 seconds minimum green
            max_green_time=10,   # 50 seconds maximum green
            detection_threshold=1,  # 1 vehicle triggers phase
            step_duration=5
        )
        controllers_to_test.append((actuated_controller, "Vehicle-Actuated"))
        
        # 4. RL Agent (if available)
        if rl_model_path and SB3_AVAILABLE and os.path.exists(rl_model_path):
            try:
                rl_agent = PPO.load(rl_model_path)
                controllers_to_test.append((rl_agent, "RL Agent (PPO)"))
                print(f"✅ Loaded RL model from: {rl_model_path}")
            except Exception as e:
                print(f"❌ Failed to load RL model: {e}")
        else:
            print("⚠️  No RL model provided or Stable Baselines3 not available")
        
        # Test all controllers
        all_results = {}
        for controller, name in controllers_to_test:
            results = self.test_controller(controller, name, num_episodes, episode_length)
            all_results[name] = results
        
        # Store results
        self.results = all_results
        
        # Print comparison summary
        self.print_comparison_summary()
        
        return all_results
    
    def print_comparison_summary(self):
        """Print a formatted summary of the comparison results."""
        if not self.results:
            print("No results to display")
            return
        
        print("\n" + "=" * 60)
        print("COMPARISON SUMMARY")
        print("=" * 60)
        
        # Create comparison table
        metrics = ['avg_reward', 'avg_waiting_time', 'avg_phase_changes', 'avg_throughput']
        metric_names = ['Avg Reward', 'Avg Waiting Time (s)', 'Avg Phase Changes', 'Avg Throughput']
        
        print(f"{'Controller':<25} {'Reward':<12} {'Wait Time':<12} {'Phase Chg':<12} {'Throughput':<12}")
        print("-" * 75)
        
        for controller_name, results in self.results.items():
            print(f"{controller_name:<25} "
                  f"{results['avg_reward']:<12.2f} "
                  f"{results['avg_waiting_time']:<12.2f} "
                  f"{results['avg_phase_changes']:<12.1f} "
                  f"{results['avg_throughput']:<12.1f}")
        
        # Find best performers
        print("\nBest Performers:")
        best_reward = max(self.results.items(), key=lambda x: x[1]['avg_reward'])
        best_waiting = min(self.results.items(), key=lambda x: x[1]['avg_waiting_time'])
        
        print(f"  Highest Reward: {best_reward[0]} ({best_reward[1]['avg_reward']:.2f})")
        print(f"  Lowest Waiting Time: {best_waiting[0]} ({best_waiting[1]['avg_waiting_time']:.2f}s)")
    
    def save_results(self, filename: str = None):
        """Save comparison results to JSON file."""
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"results/phase_1_multi_scale_validation/data/controller_comparison_{timestamp}.json"
        
        # Create results directory if it doesn't exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"Results saved to: {filename}")
        return filename


def main():
    """Main function to run the comparison."""
    # Initialize comparison
    comparison = TrafficControllerComparison()
    
    # Look for trained RL model
    model_path = None
    models_dir = "models"
    if os.path.exists(models_dir):
        model_files = [f for f in os.listdir(models_dir) if f.endswith('.zip')]
        if model_files:
            # Use the most recent model file
            model_files.sort()
            model_path = os.path.join(models_dir, model_files[-1])
            print(f"Found trained model: {model_path}")
    
    # Run comparison
    results = comparison.run_comparison(
        rl_model_path=model_path,
        num_episodes=5,
        episode_length=120  # 10 minutes per episode
    )
    
    # Save results
    comparison.save_results()
    
    print("\n✅ Comparison completed!")


if __name__ == "__main__":
    main() 