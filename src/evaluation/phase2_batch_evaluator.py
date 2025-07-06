"""
Phase 2 Batch Evaluation System

phase2_batch_evaluator.py runs 500+ varied traffic scenarios to validate that RL agent
improvements hold across diverse real-world conditions. Provides statistical
confidence intervals and worst-case (maximum wait time) validation.

Pretty similar code to phase1_batch_evaluator.py, but with some extra metrics for worst-case validation.

"""

import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
import multiprocessing as mp
from pathlib import Path
import pickle
import time
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

try:
    from src.utils.traffic_scenario_generator import TrafficScenarioGenerator
    from src.environments.enhanced_sumo_env import EnhancedSumoTrafficEnv
    from src.agents.fixed_time_controller import FixedTimeController, AdaptiveFixedTimeController, ActuatedController
    from stable_baselines3 import PPO
except ModuleNotFoundError:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    from src.utils.traffic_scenario_generator import TrafficScenarioGenerator
    from src.environments.enhanced_sumo_env import EnhancedSumoTrafficEnv
    from src.agents.fixed_time_controller import FixedTimeController, AdaptiveFixedTimeController, ActuatedController
    from stable_baselines3 import PPO


class Phase2BatchEvaluator:
    """
    Comprehensive batch evaluation system for Phase 2 stochastic validation.
    
    Runs multiple controllers across hundreds of varied traffic scenarios
    to validate performance improvements with statistical confidence.
    """
    
    def __init__(self, 
                 num_scenarios: int = 500,
                 episode_length: int = 300,
                 results_dir: str = "results/phase2_stochastic_validation",
                 parallel_processes: int = 2):
        """
        Initialize Phase 2 batch evaluator.
        
        Args:
            num_scenarios: Number of traffic scenarios to generate and test
            episode_length: Length of each episode in seconds
            results_dir: Directory to save results
            parallel_processes: Number of parallel processes (careful with SUMO!)
        """
        self.num_scenarios = num_scenarios
        self.episode_length = episode_length
        self.results_dir = Path(results_dir)
        self.parallel_processes = parallel_processes
        
        # Create results directory
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.scenario_generator = TrafficScenarioGenerator()
        self.controllers = {}
        self.scenarios = []
        self.results = {}
        
        # Statistical confidence settings
        self.confidence_level = 0.95  # 95% confidence intervals
        
        print(f"   Phase 2 Batch Evaluator Initialized:")
        print(f"   Scenarios to test: {num_scenarios}")
        print(f"   Episode length: {episode_length}s")
        print(f"   Parallel processes: {parallel_processes}")
        print(f"   Results directory: {results_dir}")
        print(f"   Confidence level: {self.confidence_level * 100}%")
    
    def generate_scenarios(self) -> List[Dict[str, Any]]:
        """
        Generate diverse traffic scenarios for testing.
        
        Returns:
            List of scenario configurations
        """
        print(f"\n   Generating {self.num_scenarios} Traffic Scenarios...")
        start_time = time.time()
        
        # Generate scenarios with varied patterns
        self.scenarios = self.scenario_generator.generate_scenario_batch(
            num_scenarios=self.num_scenarios,
            simulation_duration=self.episode_length
        )
        
        generation_time = time.time() - start_time
        
        # Analyze scenario diversity
        pattern_counts = {}
        for scenario in self.scenarios:
            pattern = scenario['pattern_name']
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
        
        print(f"✅ Scenario generation completed in {generation_time:.1f}s")
        print(f"Scenario diversity:")
        for pattern, count in sorted(pattern_counts.items()):
            percentage = (count / self.num_scenarios) * 100
            print(f"   {pattern}: {count} scenarios ({percentage:.1f}%)")
        
        return self.scenarios
    
    def setup_controllers(self, trained_model_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Setup different traffic controllers for comparison.
        
        Args:
            trained_model_path: Path to trained RL model (if None, trains new one)
            
        Returns:
            Dictionary of controller instances
        """
        print(f"\nSetting up Controllers...")
        
        self.controllers = {
            "fixed_time": {
                "name": "Fixed-Time Controller",
                "description": "Traditional traffic lights with rigid timing",
                "type": "baseline"
            },
            "adaptive_fixed": {
                "name": "Adaptive Fixed-Time Controller", 
                "description": "Pre-optimized timing patterns",
                "type": "baseline"
            },
            "actuated": {
                "name": "Vehicle-Actuated Controller",
                "description": "Industry-standard technology (current practice)",
                "type": "industry_standard"
            },
            "rl_agent": {
                "name": "RL Agent (PPO)",
                "description": "AI-powered adaptive controller",
                "type": "rl_agent",
                "model_path": trained_model_path
            }
        }
        
        print("✅ Controllers configured:")
        for controller_id, info in self.controllers.items():
            print(f"   {info['name']} ({info['type']})")
        
        return self.controllers
    
    def run_single_scenario_evaluation(self, 
                                     scenario_config: Dict[str, Any],
                                     controller_id: str,
                                     scenario_index: int) -> Dict[str, Any]:
        """
        Run a single scenario with a specific controller.
        
        Args:
            scenario_config: Scenario configuration from generator
            controller_id: ID of controller to test
            scenario_index: Index of scenario for tracking
            
        Returns:
            Results dictionary with performance metrics
        """
        try:
            # Create enhanced environment with scenario
            env = EnhancedSumoTrafficEnv(
                sumo_config_file=scenario_config['config_file'],
                episode_length=self.episode_length,
                track_detailed_stats=True
            )
            
            # Initialize results
            results = {
                "scenario_index": scenario_index,
                "scenario_id": scenario_config['scenario_id'],
                "pattern_type": scenario_config['pattern_name'],
                "controller_id": controller_id,
                "success": False,
                "error": None
            }
            
            # Run episode based on controller type
            if controller_id == "rl_agent":
                # Load trained RL model
                model_path = self.controllers[controller_id].get('model_path')
                if model_path and os.path.exists(model_path):
                    model = PPO.load(model_path)
                    results.update(self._run_rl_episode(env, model))
                else:
                    # Use trained model from Phase 1 if available
                    model_files = list(Path("models").glob("*.zip"))
                    if model_files:
                        model = PPO.load(str(model_files[0]))
                        results.update(self._run_rl_episode(env, model))
                    else:
                        raise FileNotFoundError("No trained RL model found")
            
            elif controller_id == "fixed_time":
                results.update(self._run_fixed_time_episode(env))
            
            elif controller_id == "adaptive_fixed":
                results.update(self._run_adaptive_fixed_episode(env, scenario_config))
            
            elif controller_id == "actuated":
                results.update(self._run_actuated_episode(env))
            
            results["success"] = True
            env.close()
            
            return results
            
        except Exception as e:
            return {
                "scenario_index": scenario_index,
                "scenario_id": scenario_config.get('scenario_id', 'unknown'),
                "pattern_type": scenario_config.get('pattern_name', 'unknown'),
                "controller_id": controller_id,
                "success": False,
                "error": str(e),
                "avg_waiting_time": 9999,  # High penalty for errors
                "max_waiting_time": 9999,
                "total_reward": -9999,
                "phase_changes": 0,
                "vehicle_count": 0
            }
    
    def _run_rl_episode(self, env: EnhancedSumoTrafficEnv, model: PPO) -> Dict[str, Any]:
        """Run episode with RL agent."""
        obs, info = env.reset()
        total_reward = 0
        
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            if terminated or truncated:
                break
        
        # Get comprehensive episode statistics
        episode_data = env.get_detailed_episode_data()
        episode_summary = episode_data['episode_summary']
        
        return {
            "avg_waiting_time": episode_summary.get('episode_avg_waiting_time', 0),
            "max_waiting_time": episode_summary.get('episode_max_waiting_time', 0),
            "total_reward": total_reward,
            "phase_changes": episode_summary.get('episode_phase_changes', 0),
            "vehicle_count": episode_summary.get('traffic_analysis', {}).get('total_vehicle_steps', 0),
            "episode_steps": episode_summary.get('episode_total_steps', 0)
        }
    
    def _run_fixed_time_episode(self, env: EnhancedSumoTrafficEnv) -> Dict[str, Any]:
        """Run episode with fixed-time controller."""
        obs, info = env.reset()
        
        controller = FixedTimeController(
            phase_durations=[6, 6, 6, 6],
            step_duration=5
        )
        
        total_reward = 0
        
        while True:
            action = controller.get_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            if terminated or truncated:
                break
        
        episode_data = env.get_detailed_episode_data()
        episode_summary = episode_data['episode_summary']
        
        return {
            "avg_waiting_time": episode_summary.get('episode_avg_waiting_time', 0),
            "max_waiting_time": episode_summary.get('episode_max_waiting_time', 0),
            "total_reward": total_reward,
            "phase_changes": episode_summary.get('episode_phase_changes', 0),
            "vehicle_count": episode_summary.get('traffic_analysis', {}).get('total_vehicle_steps', 0),
            "episode_steps": episode_summary.get('episode_total_steps', 0)
        }
    
    def _run_adaptive_fixed_episode(self, env: EnhancedSumoTrafficEnv, scenario_config: Dict[str, Any]) -> Dict[str, Any]:
        """Run episode with adaptive fixed-time controller (optimized for traffic pattern)."""
        obs, info = env.reset()
        
        pattern = scenario_config['pattern_name']
        if 'rush_hour' in pattern:
            # Longer durations for heavy traffic
            controller = AdaptiveFixedTimeController(north_south_duration=9, east_west_duration=6)  # 45s, 30s
        elif 'light' in pattern or 'off_peak' in pattern:
            # Shorter durations for light traffic
            controller = AdaptiveFixedTimeController(north_south_duration=4, east_west_duration=3)  # 20s, 15s
        else:
            # Standard adaptive timing
            controller = AdaptiveFixedTimeController(north_south_duration=8, east_west_duration=4)  # 40s, 20s
        
        total_reward = 0
        
        while True:
            action = controller.get_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            if terminated or truncated:
                break
        
        episode_data = env.get_detailed_episode_data()
        episode_summary = episode_data['episode_summary']
        
        return {
            "avg_waiting_time": episode_summary.get('episode_avg_waiting_time', 0),
            "max_waiting_time": episode_summary.get('episode_max_waiting_time', 0),
            "total_reward": total_reward,
            "phase_changes": episode_summary.get('episode_phase_changes', 0),
            "vehicle_count": episode_summary.get('traffic_analysis', {}).get('total_vehicle_steps', 0),
            "episode_steps": episode_summary.get('episode_total_steps', 0)
        }
    
    def _run_actuated_episode(self, env: EnhancedSumoTrafficEnv) -> Dict[str, Any]:
        """Run episode with actuated controller (industry standard)."""
        obs, info = env.reset()
        
        # Use proper ActuatedController from fixed_time_controller.py
        controller = ActuatedController(
            min_green_time=3,    # 15 seconds minimum green (3 steps × 5s)
            max_green_time=12,   # 60 seconds maximum green (12 steps × 5s)
            detection_threshold=1,  # 1 vehicle triggers phase change
            step_duration=5
        )
        
        total_reward = 0
        
        while True:
            action = controller.get_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            if terminated or truncated:
                break
        
        episode_data = env.get_detailed_episode_data()
        episode_summary = episode_data['episode_summary']
        
        return {
            "avg_waiting_time": episode_summary.get('episode_avg_waiting_time', 0),
            "max_waiting_time": episode_summary.get('episode_max_waiting_time', 0),
            "total_reward": total_reward,
            "phase_changes": episode_summary.get('episode_phase_changes', 0),
            "vehicle_count": episode_summary.get('traffic_analysis', {}).get('total_vehicle_steps', 0),
            "episode_steps": episode_summary.get('episode_total_steps', 0)
        }
    
    def run_batch_evaluation(self, trained_model_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Run complete batch evaluation across all scenarios and controllers.
        
        Args:
            trained_model_path: Path to trained RL model
            
        Returns:
            Comprehensive results dictionary
        """
        print(f"\n Starting Phase 2 Batch Evaluation")
        print(f"{'='*60}")
        start_time = time.time()
        
        # Step 1: Generate scenarios
        if not self.scenarios:
            self.generate_scenarios()
        
        # Step 2: Setup controllers
        self.setup_controllers(trained_model_path)
        
        # Step 3: Run all combinations
        print(f"\n Running evaluations...")
        print(f"   Total combinations: {len(self.scenarios)} scenarios × {len(self.controllers)} controllers = {len(self.scenarios) * len(self.controllers)} runs")
        
        all_results = []
        total_runs = len(self.scenarios) * len(self.controllers)
        completed_runs = 0
        
        for controller_id in self.controllers.keys():
            print(f"\n Testing {self.controllers[controller_id]['name']}...")
            controller_results = []
            
            for i, scenario in enumerate(self.scenarios):
                result = self.run_single_scenario_evaluation(scenario, controller_id, i)
                controller_results.append(result)
                all_results.append(result)
                completed_runs += 1
                
                # Progress update
                if (i + 1) % 50 == 0 or (i + 1) == len(self.scenarios):
                    progress = (completed_runs / total_runs) * 100
                    print(f"   Progress: {i+1}/{len(self.scenarios)} scenarios ({progress:.1f}% total)")
        
        # Step 4: Analyze results
        print(f"\nAnalyzing results...")
        analysis = self.analyze_results(all_results)
        
        # Step 5: Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.results_dir / f"phase2_batch_results_{timestamp}.json"
        summary_file = self.results_dir / f"phase2_summary_{timestamp}.csv"
        
        # Save detailed results
        with open(results_file, 'w') as f:
            json.dump({
                "metadata": {
                    "num_scenarios": self.num_scenarios,
                    "episode_length": self.episode_length,
                    "timestamp": timestamp,
                    "total_runtime": time.time() - start_time
                },
                "scenarios": self.scenarios,
                "controllers": self.controllers,
                "results": all_results,
                "analysis": analysis
            }, f, indent=2)
        
        # Save summary CSV
        self.save_summary_csv(analysis, summary_file)
        
        total_time = time.time() - start_time
        print(f"\n✅ Phase 2 Batch Evaluation Complete!")
        print(f"   Total runtime: {total_time/60:.1f} minutes")
        print(f"   Results saved to: {results_file}")
        print(f"   Summary saved to: {summary_file}")
        
        return analysis
    
    def analyze_results(self, all_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Perform comprehensive statistical analysis of results.
        
        Args:
            all_results: List of all evaluation results
            
        Returns:
            Statistical analysis dictionary
        """
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(all_results)
        successful_results = df[df['success'] == True]
        
        if len(successful_results) == 0:
            return {"error": "No successful runs to analyze"}
        
        # Group by controller
        analysis = {
            "summary": {},
            "statistical_tests": {},
            "confidence_intervals": {},
            "worst_case_analysis": {}
        }
        
        for controller_id in df['controller_id'].unique():
            controller_data = successful_results[successful_results['controller_id'] == controller_id]
            
            if len(controller_data) == 0:
                continue
            
            # Basic statistics
            analysis["summary"][controller_id] = {
                "name": self.controllers[controller_id]["name"],
                "total_runs": len(controller_data),
                "successful_runs": len(controller_data),
                "avg_waiting_time": {
                    "mean": float(controller_data['avg_waiting_time'].mean()),
                    "std": float(controller_data['avg_waiting_time'].std()),
                    "min": float(controller_data['avg_waiting_time'].min()),
                    "max": float(controller_data['avg_waiting_time'].max()),
                    "median": float(controller_data['avg_waiting_time'].median())
                },
                "max_waiting_time": {
                    "mean": float(controller_data['max_waiting_time'].mean()),
                    "std": float(controller_data['max_waiting_time'].std()),
                    "min": float(controller_data['max_waiting_time'].min()),
                    "max": float(controller_data['max_waiting_time'].max()),
                    "median": float(controller_data['max_waiting_time'].median())
                }
            }
            
            # Confidence intervals
            avg_wait_data = controller_data['avg_waiting_time'].values
            max_wait_data = controller_data['max_waiting_time'].values
            
            analysis["confidence_intervals"][controller_id] = {
                "avg_waiting_time": self._calculate_confidence_interval(avg_wait_data),
                "max_waiting_time": self._calculate_confidence_interval(max_wait_data)
            }
        
        # Comparative analysis (RL vs others)
        if 'rl_agent' in analysis["summary"] and 'actuated' in analysis["summary"]:
            rl_avg_wait = successful_results[successful_results['controller_id'] == 'rl_agent']['avg_waiting_time']
            actuated_avg_wait = successful_results[successful_results['controller_id'] == 'actuated']['avg_waiting_time']
            
            rl_max_wait = successful_results[successful_results['controller_id'] == 'rl_agent']['max_waiting_time']
            actuated_max_wait = successful_results[successful_results['controller_id'] == 'actuated']['max_waiting_time']
            
            # Statistical significance tests
            avg_t_stat, avg_p_value = stats.ttest_ind(rl_avg_wait, actuated_avg_wait)
            max_t_stat, max_p_value = stats.ttest_ind(rl_max_wait, actuated_max_wait)
            
            # Performance improvement calculations
            avg_improvement = ((actuated_avg_wait.mean() - rl_avg_wait.mean()) / actuated_avg_wait.mean()) * 100
            max_improvement = ((actuated_max_wait.mean() - rl_max_wait.mean()) / actuated_max_wait.mean()) * 100
            
            analysis["statistical_tests"]["rl_vs_actuated"] = {
                "avg_waiting_time": {
                    "t_statistic": float(avg_t_stat),
                    "p_value": float(avg_p_value),
                    "significant": bool(avg_p_value < 0.05), 
                    "improvement_percentage": float(avg_improvement)
                },
                "max_waiting_time": {
                    "t_statistic": float(max_t_stat),
                    "p_value": float(max_p_value),
                    "significant": bool(max_p_value < 0.05), 
                    "improvement_percentage": float(max_improvement)
                }
            }
        
        return analysis
    
    def _calculate_confidence_interval(self, data: np.ndarray, confidence: float = 0.95) -> Dict[str, float]:
        """Calculate confidence interval for data."""
        mean = np.mean(data)
        sem = stats.sem(data)  # Standard error of mean
        interval = stats.t.interval(confidence, len(data)-1, loc=mean, scale=sem)
        
        return {
            "mean": float(mean),
            "lower_bound": float(interval[0]),
            "upper_bound": float(interval[1]),
            "margin_of_error": float(interval[1] - mean)
        }
    
    def save_summary_csv(self, analysis: Dict[str, Any], filepath: Path):
        """Save summary statistics to CSV."""
        summary_data = []
        
        for controller_id, stats in analysis.get("summary", {}).items():
            row = {
                "Controller": stats["name"],
                "Total_Runs": stats["total_runs"],
                "Avg_Wait_Mean": stats["avg_waiting_time"]["mean"],
                "Avg_Wait_Std": stats["avg_waiting_time"]["std"],
                "Max_Wait_Mean": stats["max_waiting_time"]["mean"],
                "Max_Wait_Std": stats["max_waiting_time"]["std"]
            }
            
            # Add confidence intervals if available
            if controller_id in analysis.get("confidence_intervals", {}):
                ci = analysis["confidence_intervals"][controller_id]
                row.update({
                    "Avg_Wait_CI_Lower": ci["avg_waiting_time"]["lower_bound"],
                    "Avg_Wait_CI_Upper": ci["avg_waiting_time"]["upper_bound"],
                    "Max_Wait_CI_Lower": ci["max_waiting_time"]["lower_bound"],
                    "Max_Wait_CI_Upper": ci["max_waiting_time"]["upper_bound"]
                })
            
            summary_data.append(row)
        
        df = pd.DataFrame(summary_data)
        df.to_csv(filepath, index=False)


def main():
    """Main function for testing batch evaluator."""
    print(" Testing Phase 2 Batch Evaluator")
    print("=" * 50)
    
    # Test with small number for demonstration
    evaluator = Phase2BatchEvaluator(
        num_scenarios=10,  # Small test run
        episode_length=120,  # 2 minutes per scenario
        parallel_processes=1
    )
    
    # Run evaluation
    results = evaluator.run_batch_evaluation()
    
    print("\n Quick Results Summary:")
    if "summary" in results:
        for controller_id, stats in results["summary"].items():
            print(f"   {stats['name']}:")
            print(f"      Avg wait time: {stats['avg_waiting_time']['mean']:.3f}s")
            print(f"      Max wait time: {stats['max_waiting_time']['mean']:.3f}s")
    
    print("\n✅ Batch evaluator test completed!")


if __name__ == "__main__":
    main() 