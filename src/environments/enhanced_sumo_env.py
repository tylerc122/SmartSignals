"""
Enhanced SUMO Traffic Environment for Phase 2 Stochastic Validation

Pretty much the same as the original sumo_traffic_env.py, but with some extra metrics
tracking for worst-case validation rather than just average. Essentially just changed step() and reset() and
added _get_enhanced_reward_and_info() and _get_episode_summary() all the other functions are exactly the same, 
inherited from sumo_traffic_env.py

New features:
- Maximum wait time tracking per episode (worst-case validation)
- Enhanced metrics collection (worst-case scenarios)
- Detailed per-step statistics for analysis
- Support for varied traffic scenarios from scenario generator
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import traci
import sumolib
import os
import sys
from typing import Dict, Any, Tuple, Optional, List

try:
    from src.environments.sumo_traffic_env import SumoTrafficEnv
except ModuleNotFoundError:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    from src.environments.sumo_traffic_env import SumoTrafficEnv


class EnhancedSumoTrafficEnv(SumoTrafficEnv):
    """
    Enhanced SUMO Traffic Environment for Phase 2 validation.
    
    Extends the base SumoTrafficEnv with additional metrics tracking:
    - Maximum wait times (worst-case validation)
    - Per-step detailed statistics
    - Enhanced episode summaries
    - Support for varied traffic scenarios
    """
    
    def __init__(self, 
                 sumo_config_file="sumo_scenarios/cross_intersection.sumocfg",
                 step_duration=5,
                 episode_length=60,
                 track_detailed_stats=True):
        """
        Initialize enhanced SUMO environment with additional tracking.
        
        Args:
            sumo_config_file: Path to SUMO configuration file
            step_duration: Seconds of simulation per RL step
            episode_length: Length of episode in simulation seconds
            track_detailed_stats: Whether to collect detailed per-step statistics
        """
        super().__init__(sumo_config_file, step_duration, episode_length)
        
        # Enhanced tracking for Phase 2
        self.track_detailed_stats = track_detailed_stats
        self.reset_episode_stats()
        
        print(f"Enhanced SUMO Environment initialized:")
        print(f"   Tracking maximum wait times: ✅")
        print(f"   Detailed statistics: {'✅' if track_detailed_stats else '❌'}")
        print(f"   Phase 2 ready: ✅")
    
    def _start_sumo(self):
        """Start SUMO simulation with improved connection management for batch processing."""
        # Force close any existing connections first
        try:
            traci.close()
        except:
            pass  # Ignore if no connection exists
        
        self.is_connected = False
            
        # Use command if already set (for GUI), otherwise default to headless
        if self.sumo_cmd is None:
            self.sumo_cmd = ["sumo", "-c", self.sumo_config_file, 
                             "--no-step-log", "--no-warnings"]
        
        # For GUI mode, add extra startup time and error handling
        is_gui = "sumo-gui" in self.sumo_cmd[0]
        
        try:
            if is_gui:
                print("Starting SUMO GUI")
                import time
                time.sleep(2)  # Give GUI time to start
            
            # Start SUMO with explicit connection management
            traci.start(self.sumo_cmd)
            self.is_connected = True
            
            if is_gui:
                print("✅ SUMO GUI connected successfully!")
                # Extra pause to let GUI fully initialize
                time.sleep(1)
            
        except Exception as e:
            print(f"❌ Failed to start SUMO: {e}")
            if is_gui:
                print("XQuartz?")
            # Try to clean up before re-raising
            try:
                traci.close()
            except:
                pass
            raise
        
        # Verify traffic light exists
        try:
            if self.traffic_light_id not in traci.trafficlight.getIDList():
                available_tls = traci.trafficlight.getIDList()
                raise ValueError(f"Traffic light '{self.traffic_light_id}' not found. "
                               f"Available traffic lights: {available_tls}")
        except Exception as e:
            print(f"❌ Traffic light verification failed: {e}")
            self._close_sumo()
            raise
    
    def reset_episode_stats(self):
        """Reset all episode-level statistics tracking."""
        # Basic stats (from original environment)
        self.episode_total_waiting = 0
        self.episode_steps = 0
        
        # Enhanced stats for Phase 2
        self.max_wait_time_episode = 0.0           # Maximum wait time in episode
        self.max_wait_time_per_step = []           # Maximum wait time each step
        self.avg_wait_time_per_step = []           # Average wait time each step
        self.total_vehicles_per_step = []          # Vehicle count each step
        self.phase_changes = 0                     # Number of phase changes
        self.last_action = None                    # Track phase changes
        
        # Detailed per-step metrics (if enabled)
        if self.track_detailed_stats:
            self.detailed_step_stats = []
            self.vehicle_wait_histories = {}       # Track individual vehicle wait times
    
    def _get_enhanced_reward_and_info(self):
        """
        Calculate enhanced reward and collect comprehensive statistics.
        
        Returns:
            tuple: (reward, info_dict)
        """
        # Get all vehicles currently in simulation
        vehicle_ids = traci.vehicle.getIDList()
        
        if not vehicle_ids:
            return 0.0, {
                "total_waiting_time": 0.0,
                "max_waiting_time": 0.0,
                "avg_waiting_time": 0.0,
                "vehicle_count": 0,
                "current_phase": traci.trafficlight.getPhase(self.traffic_light_id) if self.is_connected else 0
            }
        
        # Calculate waiting time statistics
        wait_times = [traci.vehicle.getWaitingTime(vid) for vid in vehicle_ids]
        
        total_waiting = sum(wait_times)
        max_waiting = max(wait_times) if wait_times else 0.0
        avg_waiting = np.mean(wait_times) if wait_times else 0.0
        vehicle_count = len(vehicle_ids)
        
        # Update episode maximums
        self.max_wait_time_episode = max(self.max_wait_time_episode, max_waiting)
        
        # Store per-step statistics
        self.max_wait_time_per_step.append(max_waiting)
        self.avg_wait_time_per_step.append(avg_waiting)
        self.total_vehicles_per_step.append(vehicle_count)
        
        # Detailed tracking (if enabled)
        if self.track_detailed_stats:
            # Track individual vehicle histories
            for vid in vehicle_ids:
                wait_time = traci.vehicle.getWaitingTime(vid)
                if vid not in self.vehicle_wait_histories:
                    self.vehicle_wait_histories[vid] = []
                self.vehicle_wait_histories[vid].append(wait_time)
            
            # Store detailed step statistics
            step_stats = {
                "step": self.current_step // self.step_duration,
                "vehicle_count": vehicle_count,
                "total_waiting": total_waiting,
                "max_waiting": max_waiting,
                "avg_waiting": avg_waiting,
                "wait_time_distribution": {
                    "min": min(wait_times) if wait_times else 0.0,
                    "25th_percentile": np.percentile(wait_times, 25) if wait_times else 0.0,
                    "median": np.median(wait_times) if wait_times else 0.0,
                    "75th_percentile": np.percentile(wait_times, 75) if wait_times else 0.0,
                    "95th_percentile": np.percentile(wait_times, 95) if wait_times else 0.0,
                    "max": max_waiting
                },
                "current_phase": traci.trafficlight.getPhase(self.traffic_light_id)
            }
            self.detailed_step_stats.append(step_stats)
        
        # Calculate reward (negative total waiting time, same as original)
        reward = -total_waiting
        
        # Comprehensive info dictionary
        info = {
            "total_waiting_time": total_waiting,
            "max_waiting_time": max_waiting,
            "avg_waiting_time": avg_waiting,
            "vehicle_count": vehicle_count,
            "current_phase": traci.trafficlight.getPhase(self.traffic_light_id),
            
            # Episode-level maximums
            "episode_max_waiting": self.max_wait_time_episode,
            "episode_steps": len(self.max_wait_time_per_step),
            "phase_changes": self.phase_changes
        }
        
        return reward, info
    
    def step(self, action):
        """
        Enhanced step function with comprehensive statistics tracking.
        
        Args:
            action (int): Action to take
            
        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        try:
            # Track phase changes
            if self.last_action is not None and action != self.last_action:
                self.phase_changes += 1
            self.last_action = action
            
            # Apply action
            self._apply_action(action)
            
            # Run simulation for step_duration seconds
            for _ in range(self.step_duration):
                traci.simulationStep()
            
            # Get new state
            new_state = self._get_state()
            
            # Get enhanced reward and comprehensive info
            reward, info = self._get_enhanced_reward_and_info()
            
            # Update step counter
            self.current_step += self.step_duration
            self.episode_steps += 1
            
            # Check if episode is done
            terminated = self.current_step >= self.episode_length
            truncated = False
            
            # Add episode summary to info if episode is done
            if terminated:
                info.update(self._get_episode_summary())
            
            return new_state, reward, terminated, truncated, info
            
        except Exception as e:
            print(f"❌ Error during enhanced simulation step: {e}")
            
            # Try to recover
            try:
                self._close_sumo()
                self._start_sumo()
                
                # Return safe default state
                safe_state = np.zeros(20, dtype=np.float32)
                safe_info = {
                    "error": str(e),
                    "total_waiting_time": 1000,  # High penalty for errors
                    "max_waiting_time": 1000,
                    "avg_waiting_time": 1000,
                    "vehicle_count": 0,
                    "current_phase": 0
                }
                return safe_state, -1000, True, False, safe_info
                
            except Exception as recovery_error:
                print(f"❌ Recovery failed: {recovery_error}")
                safe_state = np.zeros(20, dtype=np.float32)
                safe_info = {
                    "error": str(e),
                    "recovery_error": str(recovery_error),
                    "total_waiting_time": 1000,
                    "max_waiting_time": 1000,
                    "avg_waiting_time": 1000,
                    "vehicle_count": 0,
                    "current_phase": 0
                }
                return safe_state, -1000, True, False, safe_info
    
    def reset(self, **kwargs):
        """
        Enhanced reset function with statistics reset.
        
        Returns:
            tuple: (observation, info)
        """
        # Reset episode statistics
        self.reset_episode_stats()
        
        # Call parent reset
        observation, info = super().reset(**kwargs)
        
        # Add enhanced info
        enhanced_info = {
            "episode_max_waiting": 0.0,
            "episode_steps": 0,
            "phase_changes": 0,
            "enhanced_tracking": True
        }
        info.update(enhanced_info)
        
        return observation, info
    
    def _get_episode_summary(self) -> Dict[str, Any]:
        """
        Generate comprehensive episode summary statistics.
        
        Returns:
            Dictionary with episode-level statistics
        """
        if not self.max_wait_time_per_step:
            return {"episode_summary": "No data collected"}
        
        # Calculate episode-level statistics
        episode_summary = {
            # Core Phase 2 metrics
            "episode_max_waiting_time": self.max_wait_time_episode,
            "episode_avg_waiting_time": np.mean(self.avg_wait_time_per_step),
            "episode_total_steps": len(self.max_wait_time_per_step),
            "episode_phase_changes": self.phase_changes,
            
            # Worst-case analysis
            "worst_case_percentiles": {
                "max_waiting_95th": np.percentile(self.max_wait_time_per_step, 95),
                "max_waiting_99th": np.percentile(self.max_wait_time_per_step, 99),
                "max_waiting_absolute": max(self.max_wait_time_per_step) if self.max_wait_time_per_step else 0
            },
            
            # Traffic flow analysis
            "traffic_analysis": {
                "avg_vehicles_per_step": np.mean(self.total_vehicles_per_step),
                "max_vehicles_per_step": max(self.total_vehicles_per_step) if self.total_vehicles_per_step else 0,
                "total_vehicle_steps": sum(self.total_vehicles_per_step)
            },
            
            # Performance consistency
            "performance_consistency": {
                "max_wait_std": np.std(self.max_wait_time_per_step),
                "avg_wait_std": np.std(self.avg_wait_time_per_step),
                "max_wait_variance": np.var(self.max_wait_time_per_step)
            }
        }
        
        # Add detailed statistics if tracking enabled
        if self.track_detailed_stats and hasattr(self, 'detailed_step_stats'):
            episode_summary["detailed_stats_available"] = True
            episode_summary["total_detailed_steps"] = len(self.detailed_step_stats)
            
            # Vehicle lifecycle analysis
            if self.vehicle_wait_histories:
                vehicle_max_waits = [max(waits) for waits in self.vehicle_wait_histories.values()]
                episode_summary["vehicle_analysis"] = {
                    "total_unique_vehicles": len(self.vehicle_wait_histories),
                    "vehicle_max_wait_mean": np.mean(vehicle_max_waits),
                    "vehicle_max_wait_max": max(vehicle_max_waits) if vehicle_max_waits else 0,
                    "vehicles_with_long_waits": sum(1 for max_wait in vehicle_max_waits if max_wait > 60)  # >1 minute
                }
        
        return episode_summary
    
    def get_detailed_episode_data(self) -> Dict[str, Any]:
        """
        Get all detailed episode data for analysis.
        
        Returns:
            Dictionary with complete episode data
        """
        if not self.track_detailed_stats:
            return {"error": "Detailed tracking not enabled"}
        
        return {
            "episode_summary": self._get_episode_summary(),
            "step_by_step_stats": self.detailed_step_stats if hasattr(self, 'detailed_step_stats') else [],
            "max_wait_per_step": self.max_wait_time_per_step,
            "avg_wait_per_step": self.avg_wait_time_per_step,
            "vehicles_per_step": self.total_vehicles_per_step,
            "vehicle_wait_histories": self.vehicle_wait_histories if hasattr(self, 'vehicle_wait_histories') else {}
        }


def test_enhanced_environment():
    """Test function to demonstrate enhanced environment capabilities."""
    print("Testing Enhanced SUMO Environment")
    print("=" * 50)
    
    # Test with original scenario
    env = EnhancedSumoTrafficEnv(
        sumo_config_file="sumo_scenarios/cross_intersection.sumocfg",
        episode_length=60,  # 1 minute test
        track_detailed_stats=True
    )
    
    print("Testing basic functionality...")
    
    # Reset environment
    obs, info = env.reset()
    print(f"✅ Environment reset successful")
    print(f"   Initial info keys: {list(info.keys())}")
    
    # Run a few steps
    total_reward = 0
    for step in range(5):
        action = step % 4  # Cycle through actions
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        print(f"Step {step + 1}:")
        print(f"   Max waiting time: {info.get('max_waiting_time', 0):.2f}s")
        print(f"   Avg waiting time: {info.get('avg_waiting_time', 0):.2f}s")
        print(f"   Vehicle count: {info.get('vehicle_count', 0)}")
        
        if terminated or truncated:
            break
    
    print(f"\nEpisode Summary:")
    episode_data = env.get_detailed_episode_data()
    summary = episode_data.get('episode_summary', {})
    print(f"   Episode max waiting: {summary.get('episode_max_waiting_time', 0):.2f}s")
    print(f"   Episode avg waiting: {summary.get('episode_avg_waiting_time', 0):.2f}s")
    print(f"   Phase changes: {summary.get('episode_phase_changes', 0)}")
    
    env.close()
    print("✅ Enhanced environment test completed!")


if __name__ == "__main__":
    test_enhanced_environment() 