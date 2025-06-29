"""
This module creates a Gymnasium-compatible environment that wraps SUMO traffic simulation,
allowing RL agents to control traffic lights and learn optimal signal timing policies.
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
    Custom Gymnasium environment for traffic light control using SUMO simulation.
    
    This environment allows an RL agent to control traffic signals at an intersection
    by observing traffic state and taking actions to change signal phases.
    """
    
    def __init__(self, 
                 sumo_config_path: str,
                 episode_length: int = 300,
                 step_size: int = 5,
                 render_mode: Optional[str] = None):
        """
        Initialize the SUMO traffic environment.
        
        Args:
            sumo_config_path: Path to SUMO configuration file (.sumocfg)
            episode_length: Length of each episode in simulation seconds
            step_size: Number of simulation seconds per environment step
            render_mode: Rendering mode ('human' for GUI, None for headless)
        """
        super().__init__()
        
        # Environment parameters
        self.sumo_config_path = sumo_config_path
        self.episode_length = episode_length
        self.step_size = step_size
        self.render_mode = render_mode
        
        # SUMO connection
        self.sumo_connected = False
        self.simulation_step = 0
        
        # Traffic light ID (assuming single intersection)
        self.traffic_light_id = "B1"  # Traffic light ID from netconvert
        
        # Define action space: 4 discrete actions for 4-way intersection
        # 0: North-South Green, East-West Red
        # 1: East-West Green, North-South Red  
        # 2: North-South + Left Turn Green
        # 3: East-West + Left Turn Green
        self.action_space = spaces.Discrete(4)
        
        # Define observation space
        # State vector contains:
        # - Number of waiting vehicles per lane (4 lanes) = 4 values
        # - Average waiting time per lane (4 lanes) = 4 values  
        # - Current traffic light phase (one-hot encoded, 4 phases) = 4 values
        # Total: 12 dimensional observation space
        self.observation_space = spaces.Box(
            low=0.0,
            high=np.inf,
            shape=(12,),
            dtype=np.float32
        )
        
        # State tracking
        self.total_waiting_time = 0.0
        self.vehicle_count = 0
        
    def _start_sumo(self) -> None:
        """Start SUMO simulation with appropriate configuration."""
        if self.sumo_connected:
            return
            
        # Determine SUMO command based on render mode
        if self.render_mode == "human":
            sumo_cmd = ["sumo-gui", "-c", self.sumo_config_path]
        else:
            sumo_cmd = ["sumo", "-c", self.sumo_config_path, "--no-warnings"]
            
        # Start SUMO
        traci.start(sumo_cmd)
        self.sumo_connected = True
        
        # Auto-detect traffic light ID
        traffic_lights = traci.trafficlight.getIDList()
        if traffic_lights:
            self.traffic_light_id = traffic_lights[0]
        else:
            raise ValueError("No traffic lights found in SUMO simulation")
            
    def _get_observation(self) -> np.ndarray:
        """
        Get current state observation from SUMO simulation.
        
        Returns:
            12-dimensional state vector containing traffic information
        """
        if not self.sumo_connected:
            return np.zeros(12, dtype=np.float32)
            
        # Get lane IDs connected to the intersection
        lanes = traci.trafficlight.getControlledLanes(self.traffic_light_id)
        
        # Initialize observation components
        waiting_vehicles = np.zeros(4, dtype=np.float32)
        avg_waiting_times = np.zeros(4, dtype=np.float32)
        
        # Get traffic data for each lane (limit to 4 main lanes)
        for i, lane_id in enumerate(lanes[:4]):
            # Number of waiting vehicles (speed < 0.5 m/s)
            vehicle_ids = traci.lane.getLastStepVehicleIDs(lane_id)
            waiting_count = 0
            total_waiting_time = 0.0
            
            for vehicle_id in vehicle_ids:
                speed = traci.vehicle.getSpeed(vehicle_id)
                if speed < 0.5:  # Consider as waiting if speed < 0.5 m/s
                    waiting_count += 1
                    waiting_time = traci.vehicle.getWaitingTime(vehicle_id)
                    total_waiting_time += waiting_time
                    
            waiting_vehicles[i] = waiting_count
            avg_waiting_times[i] = total_waiting_time / max(waiting_count, 1)
            
        # Get current traffic light phase (one-hot encoded)
        current_phase = traci.trafficlight.getPhase(self.traffic_light_id)
        phase_vector = np.zeros(4, dtype=np.float32)
        if current_phase < 4:  # Ensure phase is within expected range
            phase_vector[current_phase] = 1.0
            
        # Combine all observation components
        observation = np.concatenate([
            waiting_vehicles,
            avg_waiting_times, 
            phase_vector
        ])
        
        return observation.astype(np.float32)
        
    def _calculate_reward(self) -> float:
        """
        Calculate reward based on traffic performance.
        
        Uses negative total waiting time as reward to incentivize
        minimizing vehicle delays.
        
        Returns:
            Reward value (higher is better)
        """
        if not self.sumo_connected:
            return 0.0
            
        # Calculate total waiting time across all vehicles
        total_waiting = 0.0
        vehicle_ids = traci.vehicle.getIDList()
        
        for vehicle_id in vehicle_ids:
            waiting_time = traci.vehicle.getWaitingTime(vehicle_id)
            total_waiting += waiting_time
            
        # Reward is negative waiting time (minimize waiting = maximize reward)
        reward = -total_waiting
        
        return reward
        
    def _apply_action(self, action: int) -> None:
        """
        Apply the chosen action to the traffic light.
        
        Args:
            action: Integer action representing traffic light phase
        """
        if not self.sumo_connected:
            return
            
        # Map action to SUMO traffic light phase
        # Get available phases dynamically
        try:
            logic = traci.trafficlight.getAllProgramLogics(self.traffic_light_id)
            if logic and len(logic) > 0:
                num_phases = len(logic[0].phases)
                # Map actions to available phases (cycle through them)
                target_phase = action % num_phases
            else:
                target_phase = 0  # Default to phase 0
        except:
            target_phase = 0  # Fallback to phase 0
        
        traci.trafficlight.setPhase(self.traffic_light_id, target_phase)
        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment to initial state.
        
        Returns:
            Initial observation and info dictionary
        """
        super().reset(seed=seed)
        
        # Close existing SUMO connection if any
        if self.sumo_connected:
            traci.close()
            self.sumo_connected = False
            
        # Start fresh SUMO simulation
        self._start_sumo()
        
        # Reset tracking variables
        self.simulation_step = 0
        self.total_waiting_time = 0.0
        self.vehicle_count = 0
        
        # Get initial observation
        observation = self._get_observation()
        
        info = {
            "episode_step": self.simulation_step,
            "total_vehicles": self.vehicle_count
        }
        
        return observation, info
        
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one environment step.
        
        Args:
            action: Action to take (traffic light phase)
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        if not self.sumo_connected:
            raise RuntimeError("SUMO not connected. Call reset() first.")
            
        # Apply the action (change traffic light phase)
        self._apply_action(action)
        
        # Advance SUMO simulation for step_size seconds
        for _ in range(self.step_size):
            traci.simulationStep()
            self.simulation_step += 1
            
        # Get new observation after simulation step
        observation = self._get_observation()
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Check if episode is finished
        terminated = self.simulation_step >= self.episode_length
        truncated = False  # We don't use truncation in this environment
        
        # Gather info
        info = {
            "episode_step": self.simulation_step,
            "total_vehicles": len(traci.vehicle.getIDList()) if self.sumo_connected else 0,
            "reward": reward
        }
        
        return observation, reward, terminated, truncated, info
        
    def render(self) -> None:
        """Render the environment (SUMO GUI handles this)."""
        # SUMO GUI handles rendering when render_mode="human"
        pass
        
    def close(self) -> None:
        """Clean up SUMO connection."""
        if self.sumo_connected:
            traci.close()
            self.sumo_connected = False 