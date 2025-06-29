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
        self.green_phases = [0, 2]  # Phases that allow traffic movement (not yellow)
        
        # State and action spaces
        # State: [waiting_vehicles_per_lane(8), avg_waiting_time_per_lane(8), phase_one_hot(4)]
        # 8 lanes: n2c, e2c, s2c, w2c (incoming) + c2n, c2e, c2s, c2w (outgoing)
        self.observation_space = spaces.Box(
            low=0, high=100, shape=(20,), dtype=np.float32
        )
        
        # Action: 4 discrete actions for 4 possible phases
        self.action_space = spaces.Discrete(4)
        
        # SUMO connection
        self.sumo_cmd = None
        self.is_connected = False
        
    def _start_sumo(self):
        """Start SUMO simulation."""
        if self.is_connected:
            self._close_sumo()
            
        # Use headless SUMO for training (can change to sumo-gui for visualization)
        self.sumo_cmd = ["sumo", "-c", self.sumo_config_file, 
                         "--no-step-log", "--no-warnings"]
        
        # Start SUMO
        traci.start(self.sumo_cmd)
        self.is_connected = True
        
        # Verify traffic light exists
        if self.traffic_light_id not in traci.trafficlight.getIDList():
            available_tls = traci.trafficlight.getIDList()
            raise ValueError(f"Traffic light '{self.traffic_light_id}' not found. "
                           f"Available traffic lights: {available_tls}")
    
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
        incoming_lanes = ["n2c_0", "e2c_0", "s2c_0", "w2c_0"]  # Updated for cross intersection
        outgoing_lanes = ["c2n_0", "c2e_0", "c2s_0", "c2w_0"]  # Updated for cross intersection
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
    
    def close(self):
        """Close the environment."""
        self._close_sumo()
    
    def render(self, mode='human'):
        """Render the environment (SUMO GUI)."""
        # SUMO handles its own rendering
        pass 