"""
Multi-Scale Traffic Controller Comparison

This script performs comprehensive testing of traffic controllers across different time scales:
- Short-term: 10 minutes (quick response validation)
- Medium-term: 2-4 hours (sustained performance during rush hour)
- Long-term: 8-24 hours (full daily traffic cycle validation)

This provides robust validation of controller performance across realistic scenarios.
"""

import sys
import os
import numpy as np
import pandas as pd
import time
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from environments.sumo_traffic_env import SumoTrafficEnv
from agents.fixed_time_controller import FixedTimeController, AdaptiveFixedTimeController

try:
    from stable_baselines3 import PPO
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    print("Warning: Stable Baselines3 not available. Cannot test RL agent.")


class MultiScaleTrafficComparison:
    """
    Comprehensive multi-scale comparison of traffic control strategies.
    
    Tests controllers across different time horizons to validate:
    - Quick response capability (short-term)
    - Sustained performance during peak hours (medium-term)  
    - Long-term stability across daily traffic cycles (long-term)
    """
    
    def __init__(self, sumo_config_file: str = "sumo_scenarios/cross_intersection.sumocfg"):
        self.sumo_config_file = sumo_config_file
        self.results = {}
        
        # Define test scales (in steps, where each step = 5 seconds)
        self.test_scales = {
            'short': {
                'name': 'Short-term (10 min)',
                'episode_length': 120,  # 10 minutes
                'num_episodes': 5,
                'description': 'Quick response validation'
            },
            'medium': {
                'name': 'Medium-term (2 hours)', 
                'episode_length': 1440,  # 2 hours
                'num_episodes': 3,
                'description': 'Sustained rush hour performance'
            },
            'long': {
                'name': 'Long-term (8 hours)',
                'episode_length': 5760,  # 8 hours
                'num_episodes': 2,
                'description': 'Full daily cycle validation'
            }
        }
    
    def test_controller_at_scale(self, controller, controller_name: str, 
                                scale: str, verbose: bool = True) -> Dict:
        """
        Test a single controller at a specific time scale.
        
        Args:
            controller: The controller to test (RL agent or baseline)
            controller_name: Name for reporting
            scale: Scale identifier ('short', 'medium', 'long')
            verbose: Whether to print progress
            
        Returns:
            Dictionary of performance metrics for this scale
        """
        scale_config = self.test_scales[scale]
        episode_length = scale_config['episode_length']
        num_episodes = scale_config['num_episodes']
        
        if verbose:
            print(f"\n🔍 Testing {controller_name} - {scale_config['name']}")
            print(f"   {scale_config['description']}")
            print(f"   Episodes: {num_episodes}, Length: {episode_length} steps "
                  f"({episode_length * 5 // 60:.1f} minutes)")
            print("=" * 60)
        
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
        detailed_metrics = []
        
        try:
            for episode in range(num_episodes):
                if verbose:
                    print(f"  Episode {episode + 1}/{num_episodes}")
                    start_time = time.time()
                
                # Reset environment and controller
                obs, _ = env.reset()
                if hasattr(controller, 'reset'):
                    controller.reset()
                
                episode_reward = 0
                total_waiting_time = 0
                phase_changes = 0
                last_action = None
                vehicles_passed = 0
                step_metrics = []
                
                # Progress tracking for long episodes
                checkpoint_interval = max(episode_length // 10, 1)
                
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
                    
                    # Count vehicles that passed
                    vehicles_passed += info.get('vehicles_passed', 0)
                    
                    # Store detailed metrics every checkpoint for analysis
                    if step % checkpoint_interval == 0:
                        step_metrics.append({
                            'step': step,
                            'reward': reward,
                            'waiting_time': current_waiting,
                            'action': int(action),
                            'cumulative_reward': episode_reward,
                            'phase_changes': phase_changes
                        })
                    
                    # Progress indicator for long episodes
                    if verbose and episode_length > 1000 and step % checkpoint_interval == 0:
                        progress = (step / episode_length) * 100
                        print(f"    Progress: {progress:.1f}% "
                              f"(Reward: {episode_reward:.1f}, "
                              f"Waiting: {current_waiting:.1f}s)")
                    
                    if terminated or truncated:
                        break
                
                # Store episode results
                avg_waiting = total_waiting_time / episode_length
                episode_rewards.append(episode_reward)
                episode_waiting_times.append(avg_waiting)
                episode_phase_changes.append(phase_changes)
                episode_throughput.append(vehicles_passed)
                detailed_metrics.append(step_metrics)
                
                if verbose:
                    elapsed = time.time() - start_time
                    print(f"    ✅ Episode {episode + 1} completed in {elapsed:.1f}s")
                    print(f"       Reward: {episode_reward:.1f}, "
                          f"Avg Waiting: {avg_waiting:.2f}s, "
                          f"Phase Changes: {phase_changes}, "
                          f"Throughput: {vehicles_passed}")
        
        finally:
            env.close()
        
        # Calculate comprehensive statistics
        results = {
            'controller_name': controller_name,
            'scale': scale,
            'scale_config': scale_config,
            'num_episodes': num_episodes,
            'episode_length': episode_length,
            'total_simulated_time': episode_length * num_episodes * 5,  # seconds
            
            # Core performance metrics
            'avg_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'min_reward': np.min(episode_rewards),
            'max_reward': np.max(episode_rewards),
            
            'avg_waiting_time': np.mean(episode_waiting_times),
            'std_waiting_time': np.std(episode_waiting_times),
            'min_waiting_time': np.min(episode_waiting_times),
            'max_waiting_time': np.max(episode_waiting_times),
            
            'avg_phase_changes': np.mean(episode_phase_changes),
            'std_phase_changes': np.std(episode_phase_changes),
            'min_phase_changes': np.min(episode_phase_changes),
            'max_phase_changes': np.max(episode_phase_changes),
            
            'avg_throughput': np.mean(episode_throughput),
            'std_throughput': np.std(episode_throughput),
            'total_throughput': np.sum(episode_throughput),
            
            # Raw episode data for detailed analysis
            'episode_rewards': episode_rewards,
            'episode_waiting_times': episode_waiting_times,
            'episode_phase_changes': episode_phase_changes,
            'episode_throughput': episode_throughput,
            'detailed_metrics': detailed_metrics,
            
            # Timestamp for tracking
            'test_timestamp': datetime.now().isoformat()
        }
        
        if verbose:
            print(f"\n  📊 {scale_config['name']} Results Summary:")
            print(f"     Average Reward: {results['avg_reward']:.2f} ± {results['std_reward']:.2f}")
            print(f"     Average Waiting Time: {results['avg_waiting_time']:.3f} ± {results['std_waiting_time']:.3f}s")
            print(f"     Average Phase Changes: {results['avg_phase_changes']:.1f} ± {results['std_phase_changes']:.1f}")
            print(f"     Total Throughput: {results['total_throughput']:.0f} vehicles")
            print(f"     Simulated Time: {results['total_simulated_time'] / 3600:.1f} hours")
        
        return results
    
    def run_multi_scale_comparison(self, rl_model_path: str = None, 
                                  scales_to_test: List[str] = None) -> Dict:
        """
        Run complete multi-scale comparison of all available controllers.
        
        Args:
            rl_model_path: Path to trained RL model
            scales_to_test: List of scales to test ('short', 'medium', 'long')
            
        Returns:
            Complete multi-scale comparison results
        """
        if scales_to_test is None:
            scales_to_test = ['short', 'medium', 'long']
        
        print("🚦 MULTI-SCALE TRAFFIC CONTROLLER COMPARISON")
        print("=" * 70)
        print(f"Testing scales: {[self.test_scales[s]['name'] for s in scales_to_test]}")
        print(f"SUMO config: {self.sumo_config_file}")
        print(f"Test started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Initialize controllers
        controllers_to_test = []
        
        # 1. Fixed-Time Controller (Traditional)
        fixed_controller = FixedTimeController(
            phase_durations=[6, 6, 6, 6],  # 30 seconds per phase
            step_duration=5
        )
        controllers_to_test.append((fixed_controller, "Fixed-Time"))
        
        # 2. Adaptive Fixed-Time Controller
        adaptive_controller = AdaptiveFixedTimeController(
            north_south_duration=8,  # 40 seconds for main road
            east_west_duration=4,    # 20 seconds for side road
            step_duration=5
        )
        controllers_to_test.append((adaptive_controller, "Adaptive-Fixed"))
        
        # 3. RL Agent (if available)
        if rl_model_path and SB3_AVAILABLE and os.path.exists(rl_model_path):
            try:
                rl_agent = PPO.load(rl_model_path)
                controllers_to_test.append((rl_agent, "RL-Agent"))
                print(f"✅ Loaded RL model from: {rl_model_path}")
            except Exception as e:
                print(f"❌ Failed to load RL model: {e}")
        else:
            print("⚠️  No RL model provided or Stable Baselines3 not available")
        
        # Run tests for each scale and controller
        all_results = {}
        total_tests = len(scales_to_test) * len(controllers_to_test)
        current_test = 0
        
        for scale in scales_to_test:
            all_results[scale] = {}
            
            for controller, name in controllers_to_test:
                current_test += 1
                print(f"\n🔄 Test {current_test}/{total_tests}: {name} at {scale} scale")
                
                try:
                    results = self.test_controller_at_scale(controller, name, scale)
                    all_results[scale][name] = results
                    
                except Exception as e:
                    print(f"❌ Error testing {name} at {scale} scale: {e}")
                    # Continue with other tests
                    continue
        
        # Store results
        self.results = all_results
        
        # Print comprehensive summary
        self.print_multi_scale_summary()
        
        return all_results
    
    def print_multi_scale_summary(self):
        """Print a comprehensive summary of multi-scale comparison results."""
        if not self.results:
            print("No results to display")
            return
        
        print("\n" + "=" * 70)
        print("📈 MULTI-SCALE COMPARISON SUMMARY")
        print("=" * 70)
        
        # Summary table for each scale
        for scale, scale_results in self.results.items():
            if not scale_results:
                continue
                
            scale_config = self.test_scales[scale]
            print(f"\n{scale_config['name']} - {scale_config['description']}")
            print("-" * 50)
            
            # Create comparison table
            print(f"{'Controller':<15} {'Reward':<12} {'Wait Time':<12} {'Throughput':<12}")
            print("-" * 52)
            
            for controller_name, results in scale_results.items():
                print(f"{controller_name:<15} "
                      f"{results['avg_reward']:<12.1f} "
                      f"{results['avg_waiting_time']:<12.3f} "
                      f"{results['avg_throughput']:<12.1f}")
        
        # Cross-scale analysis
        print(f"\n🏆 BEST PERFORMERS ACROSS SCALES")
        print("-" * 40)
        
        for scale, scale_results in self.results.items():
            if not scale_results:
                continue
                
            # Find best performer for this scale
            best_reward = max(scale_results.items(), key=lambda x: x[1]['avg_reward'])
            best_waiting = min(scale_results.items(), key=lambda x: x[1]['avg_waiting_time'])
            
            print(f"{self.test_scales[scale]['name']}:")
            print(f"  Highest Reward: {best_reward[0]} ({best_reward[1]['avg_reward']:.1f})")
            print(f"  Lowest Wait Time: {best_waiting[0]} ({best_waiting[1]['avg_waiting_time']:.3f}s)")
    
    def save_results(self, filename: str = None, include_detailed: bool = True):
        """
        Save multi-scale comparison results to JSON file.
        
        Args:
            filename: Custom filename (auto-generated if None)
            include_detailed: Whether to include detailed step-by-step metrics
        """
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"results/phase_1_multi_scale_validation/data/multi_scale_comparison_{timestamp}.json"
        
        # Create results directory if it doesn't exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Prepare results for saving
        save_data = {
            'test_metadata': {
                'test_type': 'multi_scale_comparison',
                'test_timestamp': datetime.now().isoformat(),
                'scales_tested': list(self.results.keys()),
                'sumo_config': self.sumo_config_file
            },
            'results': self.results
        }
        
        # Optionally remove detailed metrics to reduce file size
        if not include_detailed:
            for scale_results in save_data['results'].values():
                for controller_results in scale_results.values():
                    if 'detailed_metrics' in controller_results:
                        del controller_results['detailed_metrics']
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            return obj
        
        save_data = convert_numpy_types(save_data)
        
        with open(filename, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        print(f"📁 Results saved to: {filename}")
        
        # Also save a summary CSV for easy analysis
        self.save_summary_csv(filename.replace('.json', '_summary.csv'))
        
        return filename
    
    def save_summary_csv(self, filename: str):
        """Save a summary CSV with key metrics across all scales."""
        summary_data = []
        
        for scale, scale_results in self.results.items():
            for controller_name, results in scale_results.items():
                summary_data.append({
                    'Scale': self.test_scales[scale]['name'],
                    'Controller': controller_name,
                    'Simulated_Hours': results['total_simulated_time'] / 3600,
                    'Avg_Reward': results['avg_reward'],
                    'Avg_Waiting_Time': results['avg_waiting_time'],
                    'Total_Throughput': results['total_throughput'],
                    'Avg_Phase_Changes': results['avg_phase_changes'],
                    'Std_Waiting_Time': results['std_waiting_time']
                })
        
        df = pd.DataFrame(summary_data)
        df.to_csv(filename, index=False)
        print(f"📊 Summary CSV saved to: {filename}")


def main():
    """Main function to run the multi-scale comparison."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Multi-scale traffic controller comparison')
    parser.add_argument('--scales', nargs='+', choices=['short', 'medium', 'long'], 
                       default=['short', 'medium', 'long'],
                       help='Scales to test (default: all)')
    parser.add_argument('--model', type=str, default=None,
                       help='Path to RL model (auto-detect if not specified)')
    
    args = parser.parse_args()
    
    # Initialize comparison
    comparison = MultiScaleTrafficComparison()
    
    # Look for trained RL model if not specified
    model_path = args.model
    if model_path is None:
        models_dir = "models"
        if os.path.exists(models_dir):
            model_files = [f for f in os.listdir(models_dir) if f.endswith('.zip')]
            if model_files:
                # Use the most recent model file
                model_files.sort()
                model_path = os.path.join(models_dir, model_files[-1])
                print(f"Auto-detected trained model: {model_path}")
    
    # Run multi-scale comparison
    try:
        results = comparison.run_multi_scale_comparison(
            rl_model_path=model_path,
            scales_to_test=args.scales
        )
        
        # Save results
        comparison.save_results()
        
        print(f"\n✅ Multi-scale comparison completed!")
        print(f"   Tested scales: {args.scales}")
        print(f"   Total simulated time: Multiple hours across all tests")
        
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()