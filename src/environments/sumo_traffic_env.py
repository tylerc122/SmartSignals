"""
Creates a Gymnasium-compatible environment that wraps SUMO traffic simulation,
this allows RL agents to control traffic lights and learn optimal signal timing policies,
the heart of the project.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import traci
import sumolib
import os
from typing import Dict, Any, Tuple, Optional


class SumoTrafficEnv(gym.Env):
    """
    SUMO Traffic Signal Control Environment for Reinforcement Learning
    
    This environment bridges SUMO traffic simulation with RL algorithms.
    The agent controls traffic light phases at an intersection to minimize
    vehicle waiting times.
    """
    
    def __init__(self, 
                 sumo_config_file="sumo_scenarios/cross_intersection.sumocfg",
                 step_duration=5,
                 episode_length=60):
        """
        Initialize the SUMO traffic environment.
        
        Args:
            sumo_config_file: Path to SUMO configuration file
            step_duration: Seconds of simulation per RL step
            episode_length: Length of episode in simulation seconds
        """
        super().__init__()
        
        # Environment parameters
        self.sumo_config_file = sumo_config_file
        self.step_duration = step_duration
        self.episode_length = episode_length
        self.current_step = 0
        
        # Traffic light configuration (updated for cross intersection)
        self.traffic_light_id = "center"  # Updated for our new intersection
        # Phases that allow traffic movement (not yellow)
        # 0 -> north-south green east-west red, 
        # 2 -> east-west green north-south red
        # Phases 1 & 3 are yellow transition phases
        # This tells agent 0 and 2 are productive:
        self.green_phases = [0, 2]  
        
        # State and action spaces
        # n2c_0 = north to center lane
        # e2c_0 = east to center lane
        # s2c_0 = south to center lane
        # w2c_0 = west to center lane
        # c2n_0 = center to north lane
        # c2e_0 = center to east lane
        # c2s_0 = center to south lane
        # c2w_0 = center to west lane
        # 8 lanes in total: n2c, e2c, s2c, w2c (incoming) + c2n, c2e, c2s, c2w (outgoing)
        # 20 dimenstional state vector that our agent receives every step, detailing:
        # - waiting vehicles per lane
        # - average waiting time per lane
        # - current traffic light phase (4 values with one-hot encoded)
        # i.e -> [waiting_vehicles_per_lane(8), avg_waiting_time_per_lane(8), phase_one_hot(4)]
        # Essentially, every simulated 5 seconds, the agent receives input along the lines of:
        # "There are 4 cars waiting north-to-center with average wait of 12 seconds"
        # "There are 0 cars waiting east to center"
        # "Current phase is 0 i.e north-south green, east-west red" ETC.
        # E.G:
        """
  obs = [3, 1, 0, 2,    # vehicles waiting (N,E,S,W incoming)
        0, 0, 0, 0,    # vehicles waiting (N,E,S,W outgoing)  
        12, 5, 0, 8,   # waiting times (N,E,S,W incoming)
        0, 0, 0, 0,    # waiting times (N,E,S,W outgoing)
        1, 0, 0, 0]    # phase one-hot: Phase 0 is active i.e north-south green, east-west red
        """
        # Based on the state vector, agent will decide to change the traffic light phase to 
        self.observation_space = spaces.Box(
            low=0, high=100, shape=(20,), dtype=np.float32
        )
        
        # Action: 4 discrete actions for 4 possible light phases
        self.action_space = spaces.Discrete(4)
        
        # SUMO connection
        self.sumo_cmd = None
        self.is_connected = False
        
    def _start_sumo(self):
        """Start SUMO simulation."""
        if self.is_connected:
            self._close_sumo()
            
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
            
            # Start SUMO
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
    
    def _close_sumo(self):
        """Close SUMO simulation."""
        if self.is_connected:
            traci.close()
            self.is_connected = False
    
    def _get_state(self):
        """
        Get current state of the intersection.
        
        Returns:
            np.array: State vector containing vehicle counts, waiting times, and phase info
        """
        # Get all lanes connected to the intersection
        incoming_lanes = ["n2c_0", "e2c_0", "s2c_0", "w2c_0"]
        outgoing_lanes = ["c2n_0", "c2e_0", "c2s_0", "c2w_0"]
        all_lanes = incoming_lanes + outgoing_lanes
        
        # Vehicle counts per lane
        vehicle_counts = []
        waiting_times = []
        
        for lane in all_lanes:
            # Number of waiting vehicles
            waiting_vehicles = traci.lane.getLastStepHaltingNumber(lane)
            vehicle_counts.append(waiting_vehicles)
            
            # Average waiting time
            vehicle_ids = traci.lane.getLastStepVehicleIDs(lane)
            if vehicle_ids:
                avg_waiting_time = np.mean([traci.vehicle.getWaitingTime(vid) for vid in vehicle_ids])
            else:
                avg_waiting_time = 0.0
            waiting_times.append(avg_waiting_time)
        
        # Current traffic light phase (one-hot encoded)
        current_phase = traci.trafficlight.getPhase(self.traffic_light_id)
        phase_one_hot = np.zeros(4)
        phase_one_hot[current_phase] = 1.0
        
        # Combine all state information
        state = np.array(vehicle_counts + waiting_times + phase_one_hot.tolist(), dtype=np.float32)
        
        return state
    
    def _get_reward(self):
        """
        Calculate reward based on traffic efficiency.
        
        Returns:
            float: Reward value (negative waiting time)
        """
        # Get all vehicles in the simulation
        vehicle_ids = traci.vehicle.getIDList()
        
        # Calculate total waiting time
        total_waiting_time = sum(traci.vehicle.getWaitingTime(vid) for vid in vehicle_ids)
        
        # Reward is negative waiting time (minimize waiting)
        reward = -total_waiting_time
        
        return reward
    
    def _apply_action(self, action):
        """
        Apply traffic light action.
        
        Args:
            action (int): Action to take (0-3 for phases)
        """
        # Get current number of phases
        num_phases = len(traci.trafficlight.getAllProgramLogics(self.traffic_light_id)[0].phases)
        
        # Map action to valid phase (handle case where action > num_phases)
        target_phase = action % num_phases
        
        # Set traffic light phase
        traci.trafficlight.setPhase(self.traffic_light_id, target_phase)
    
    def reset(self, **kwargs):
        """
        Reset the environment.
        
        Returns:
            tuple: (observation, info)
        """
        # Start new SUMO simulation
        self._start_sumo()
        
        # Reset episode tracking
        self.current_step = 0
        
        # Get initial state
        initial_state = self._get_state()
        
        return initial_state, {}
    
    def step(self, action):
        """
        Execute one environment step.
        
        Args:
            action (int): Action to take
            
        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        try:
            # Apply action
            self._apply_action(action)
            
            # Run simulation for step_duration seconds
            for _ in range(self.step_duration):
                traci.simulationStep()
            
            # Get new state and reward
            new_state = self._get_state()
            reward = self._get_reward()
            
            # Update step counter
            self.current_step += self.step_duration
            
            # Check if episode is done
            terminated = self.current_step >= self.episode_length
            truncated = False
            
            info = {
                "step": self.current_step,
                "total_waiting_time": -reward,
                "current_phase": traci.trafficlight.getPhase(self.traffic_light_id)
            }
            
            return new_state, reward, terminated, truncated, info
            
        except Exception as e:
            print(f"❌ Error during simulation step: {e}")
            print("🔄 Attempting to recover...")
            
            # Try to reconnect
            try:
                self._close_sumo()
                self._start_sumo()
                
                # Return safe default state
                safe_state = np.zeros(20, dtype=np.float32)
                return safe_state, -1000, True, False, {"error": str(e)}
                
            except Exception as recovery_error:
                print(f"❌ Recovery failed: {recovery_error}")
                # Return safe default and terminate
                safe_state = np.zeros(20, dtype=np.float32)
                return safe_state, -1000, True, False, {"error": str(e), "recovery_error": str(recovery_error)}
    
    def close(self):
        """Close the environment."""
        self._close_sumo()
    
    def render(self, mode='human'):
        """Render the environment (SUMO GUI)."""
        # SUMO handles its own rendering
        pass 