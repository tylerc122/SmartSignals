"""
multi_sumo_traffic_env.py extends the single intersection environment to handle multiple traffic lights in a grid network.
This allows for multi-agent reinforcement learning for coordinated traffic signal control (holy cool as hell).
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import traci
import sumolib
import os
import sys
from typing import Dict, Any, Tuple, Optional, List
from collections import defaultdict

try:
    from src.environments.enhanced_sumo_env import EnhancedSumoTrafficEnv
except ModuleNotFoundError:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    from src.environments.enhanced_sumo_env import EnhancedSumoTrafficEnv


class MultiSumoTrafficEnv(gym.Env):
    """
    Multi-Intersection SUMO Traffic Environment
    
    This environment manages multiple traffic lights in a grid network,
    supporting either independent control or coordinated control strategies.
    
    The environment can operate in different modes:
    1. Independent mode: Each traffic light as a separate agent
    2. Centralized mode: One agent controlling all traffic lights
    3. Neighbor-aware mode: Traffic lights share state with neighbors
    
    Args:
        sumo_config_file: Path to SUMO configuration file for grid network
        grid_size: Tuple (rows, cols) defining the grid dimensions
        step_duration: Duration of each simulation step in seconds
        episode_length: Length of episode in steps
        mode: Control mode ("independent", "centralized", or "neighbor_aware")
        communication_range: How many intersections away agents can communicate
    """
    
    def __init__(self, 
                 sumo_config_file="sumo_scenarios/grid_4x4_tl/grid_4x4.sumocfg",
                 grid_size=(4, 4),  # Default to 4x4 grid 
                 step_duration=5,
                 episode_length=60,
                 mode="independent",
                 communication_range=1,
                 track_detailed_stats=True):
        """Initialize the multi-intersection environment."""
        super().__init__()
        
        # Environment parameters
        self.sumo_config_file = sumo_config_file
        self.grid_size = grid_size
        self.step_duration = step_duration
        self.episode_length = episode_length
        self.current_step = 0
        self.mode = mode
        self.communication_range = communication_range
        self.track_detailed_stats = track_detailed_stats
        
        # SUMO connection
        self.sumo_cmd = None
        self.is_connected = False
        
        # Build grid network coordinates and identify all traffic lights
        self.rows, self.cols = grid_size
        self.traffic_light_ids = self._get_traffic_light_ids()
        self.num_traffic_lights = len(self.traffic_light_ids)
        
        print(f"Multi-Intersection Environment initialized with {self.num_traffic_lights} traffic lights")
        print(f"Grid size: {self.rows}x{self.cols}")
        print(f"Control mode: {self.mode}")
        
        # Create traffic light ID to grid position mapping
        self.tl_to_position = self._create_position_mapping()
        
        # Create adjacency matrix for the grid
        self.adjacency_matrix = self._create_adjacency_matrix()
        
        # Setup observation and action spaces based on mode
        self._setup_spaces()
        
        # Initialize metrics tracking
        self.global_metrics = {
            "total_waiting_time": 0,
            "max_waiting_time": 0,
            "vehicles_processed": 0,
            "emergency_vehicle_delay": 0,
        }
        
        # Per-intersection metrics
        self.intersection_metrics = {tl_id: {} for tl_id in self.traffic_light_ids}
        
    def _setup_spaces(self):
        """Setting up observations & action spaces based on whatever control mode we're in"""
        # Base state dimensions for a single intersection
        single_intersection_dims = 20  # Same as original environment
        
        if self.mode == "independent":
            # Each agent has its own observation and action space
            self.observation_space = spaces.Box(
                low=0, high=100, shape=(single_intersection_dims,), dtype=np.float32
            )
            # Same 4 actions as single intersection
            self.action_space = spaces.Discrete(4)
            
        elif self.mode == "centralized":
            # One agent controls all intersections
            self.observation_space = spaces.Box(
                low=0, high=100, 
                shape=(single_intersection_dims * self.num_traffic_lights,), 
                dtype=np.float32
            )
            # One action per traffic light, each with 4 options
            self.action_space = spaces.MultiDiscrete([4] * self.num_traffic_lights)
            
        elif self.mode == "neighbor_aware":
            # Each intersection has its own state plus neighbor states
            # For a communication range of 1, each intersection has up to 4 neighbors
            max_neighbors = min(4 * self.communication_range, self.num_traffic_lights - 1)
            
            # State includes own state plus partial states from neighbors
            # For neighbors, we only include vehicle counts and waiting times (16 values)
            # not the phase information
            neighbor_dims = 16 * max_neighbors
            
            self.observation_space = spaces.Box(
                low=0, high=100, 
                shape=(single_intersection_dims + neighbor_dims,), 
                dtype=np.float32
            )
            # Same 4 actions as single intersection
            self.action_space = spaces.Discrete(4)
            
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")
    
    def _get_traffic_light_ids(self):
        """Get all traffic light IDs from the grid network."""
        # We need to temporarily start SUMO to get the traffic light IDs
        temp_cmd = ["sumo", "-c", self.sumo_config_file, "--no-step-log", "--no-warnings"]
        
        try:
            traci.start(temp_cmd)
            tl_ids = traci.trafficlight.getIDList()
            traci.close()
            print(f"Found {len(tl_ids)} traffic lights in network: {tl_ids[:5]}...")
            return tl_ids
        except Exception as e:
            print(f"Error getting traffic light IDs: {e}")
            # If we can't get actual IDs, generate placeholder IDs based on grid dimensions
            tl_ids = []
            for i in range(1, self.rows+1):
                for j in range(1, self.cols+1):
                    tl_ids.append(f"c{i}{j}")
            
            print(f"Generated {len(tl_ids)} placeholder traffic light IDs: {tl_ids[:5]}...")
            return tl_ids
    
    def _create_position_mapping(self):
        """Map traffic light IDs to grid positions."""
        tl_to_position = {}
        
        # For the 4x4 network, traffic lights are named like 'c11', 'c12', etc.
        # where the first digit is row and second digit is column
        for tl_id in self.traffic_light_ids:
            # Try to parse position from ID for our naming convention (cij where i=row, j=column)
            if tl_id.startswith('c') and len(tl_id) == 3:
                try:
                    row = int(tl_id[1]) - 1  # Convert to 0-based index
                    col = int(tl_id[2]) - 1  # Convert to 0-based index
                    tl_to_position[tl_id] = (row, col)
                    continue
                except (ValueError, IndexError):
                    pass
            
            # For traffic lights with different naming conventions,
            # use their index to estimate a grid position
            idx = self.traffic_light_ids.index(tl_id)
            
            # For a typical grid, estimate position based on index
            # Assuming traffic lights are ordered row by row
            if self.rows > 0 and self.cols > 0:
                row = idx // self.cols
                col = idx % self.cols
            else:
                # Fallback for unknown grid size
                grid_dim = int(np.ceil(np.sqrt(len(self.traffic_light_ids))))
                row = idx // grid_dim
                col = idx % grid_dim
                
            tl_to_position[tl_id] = (row, col)
            
        return tl_to_position
    
    def _create_adjacency_matrix(self):
        """Create adjacency matrix for the grid network."""
        num_tl = len(self.traffic_light_ids)
        adjacency = np.zeros((num_tl, num_tl), dtype=bool)
        
        # Two intersections are adjacent if they are one grid cell apart
        for i, tl_id1 in enumerate(self.traffic_light_ids):
            pos1 = self.tl_to_position[tl_id1]
            
            for j, tl_id2 in enumerate(self.traffic_light_ids):
                if i == j:
                    continue
                    
                pos2 = self.tl_to_position[tl_id2]
                
                # Manhattan distance for grid adjacency Prof. Taylor would be proud
                distance = abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
                
                # Adjacent if within communication range
                if distance <= self.communication_range:
                    adjacency[i, j] = True
        
        return adjacency
    
    def _start_sumo(self):
        """Start SUMO simulation."""
        # Force close any existing connections first
        try:
            traci.close()
        except:
            pass  # Ignore if no connection exists
        
        self.is_connected = False
            
        # Use command if already set (for GUI), otherwise default to headless, not planning on using GUI for now
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
                print("SUMO GUI connected successfully")
                # Extra pause to let GUI fully initialize
                time.sleep(1)
            
            # Verify traffic lights exist
            self._verify_traffic_lights()
            
        except Exception as e:
            print(f"Failed to start SUMO: {e}")
            if is_gui:
                print("XQuartz?")
            try:
                traci.close()
            except:
                pass
            raise
    
    def _verify_traffic_lights(self):
        """Verify traffic lights in the simulation."""
        # Get all available traffic lights in the simulation
        available_tl_ids = traci.trafficlight.getIDList()
        
        if not available_tl_ids:
            raise ValueError("No traffic lights found in the simulation!")
        
        # Use all available traffic lights
        self.traffic_light_ids = available_tl_ids
        self.num_traffic_lights = len(self.traffic_light_ids)
        
        print(f"Using {self.num_traffic_lights} traffic lights: {self.traffic_light_ids[:5]}...")
        
        # Create a new position mapping based on traffic light IDs
        self.tl_to_position = self._create_position_mapping()
        
        print(f"Verified {self.num_traffic_lights} traffic lights in simulation")
    
    def _close_sumo(self):
        """Close SUMO simulation."""
        if self.is_connected:
            traci.close()
            self.is_connected = False
    
    def _get_intersection_state(self, tl_id):
        """
        Get state of a specific intersection based on traffic light ID.
        
        Args:
            tl_id: Traffic light ID in the network
            
        Returns:
            np.array: State vector for the intersection
        """
        # Get incoming and outgoing lanes for this traffic light
        incoming_lanes = []
        outgoing_lanes = []
        
        try:
            # Get links controlled by this traffic light
            controlled_links = traci.trafficlight.getControlledLinks(tl_id)
            
            # Each link is (fromLane, toLane, viaLane)
            for link_tuple in controlled_links:
                for link in link_tuple:
                    if link:  # Some links might be empty
                        from_lane, to_lane, _ = link
                        if from_lane not in incoming_lanes:
                            incoming_lanes.append(from_lane)
                        if to_lane not in outgoing_lanes:
                            outgoing_lanes.append(to_lane)
        except Exception as e:
            print(f"Error getting controlled links for {tl_id}: {e}")
            # Try alternative approaches if getting controlled links fails
        
        # If we couldn't get lanes from controlled links, try another approach
        if not incoming_lanes or not outgoing_lanes:
            try:
                # For traffic lights with cXY format, use X,Y to find connected lanes
                if tl_id.startswith('c') and len(tl_id) == 3:
                    # Get all lanes and filter by traffic light ID pattern
                    all_lanes = traci.lane.getIDList()
                    for lane in all_lanes:
                        if tl_id in lane:
                            # Try to determine if incoming or outgoing based on pattern
                            lane_parts = lane.split('_')
                            if len(lane_parts) >= 2:
                                # Lane ID format might be like "c11c12_0" (from c11 to c12)
                                if lane.startswith(tl_id):
                                    # This lane goes from our TL to somewhere else
                                    outgoing_lanes.append(lane)
                                elif lane.endswith(tl_id) or tl_id in lane:
                                    # This lane comes to our TL from somewhere else
                                    incoming_lanes.append(lane)
            except Exception as e:
                print(f"Error finding lanes for {tl_id}: {e}")
        
        # Limit to maximum 4 lanes in each direction for consistent state size
        incoming_lanes = incoming_lanes[:4] if incoming_lanes else []
        outgoing_lanes = outgoing_lanes[:4] if outgoing_lanes else []
        
        # Pad with dummy lanes if fewer than 4
        while len(incoming_lanes) < 4:
            incoming_lanes.append(f"dummy_in_{len(incoming_lanes)}")
        while len(outgoing_lanes) < 4:
            outgoing_lanes.append(f"dummy_out_{len(outgoing_lanes)}")
        
        # Get vehicle counts and waiting times
        vehicle_counts = []
        waiting_times = []
        
        for lane in incoming_lanes + outgoing_lanes:
            if lane.startswith("dummy"):
                # Dummy lanes have no vehicles or waiting time
                vehicle_counts.append(0)
                waiting_times.append(0)
            else:
                # Get real lane data
                try:
                    waiting_vehicles = traci.lane.getLastStepHaltingNumber(lane)
                    vehicle_counts.append(waiting_vehicles)
                    
                    vehicle_ids = traci.lane.getLastStepVehicleIDs(lane)
                    if vehicle_ids:
                        avg_waiting_time = np.mean([traci.vehicle.getWaitingTime(vid) for vid in vehicle_ids])
                    else:
                        avg_waiting_time = 0.0
                    waiting_times.append(avg_waiting_time)
                except traci.exceptions.TraCIException:
                    # Handle case where lane doesn't exist
                    vehicle_counts.append(0)
                    waiting_times.append(0)
        
        # Get current traffic light phase
        try:
            current_phase = traci.trafficlight.getPhase(tl_id)
            # Create one-hot encoding (assuming max 4 phases)
            phase_one_hot = np.zeros(4)
            phase_one_hot[current_phase % 4] = 1.0
        except:
            # Default to phase 0 if there's an error
            phase_one_hot = np.zeros(4)
            phase_one_hot[0] = 1.0
        
        # Combine all state information
        state = np.array(vehicle_counts + waiting_times + phase_one_hot.tolist(), dtype=np.float32)
        
        return state
    
    def _get_neighbor_states(self, tl_id):
        """
        Get partial states of neighboring intersections.
        
        Args:
            tl_id: Traffic light ID
            
        Returns:
            np.array: Combined partial states of neighbors
        """
        tl_idx = self.traffic_light_ids.index(tl_id)
        neighbor_states = []
        
        for i, is_neighbor in enumerate(self.adjacency_matrix[tl_idx]):
            if is_neighbor:
                neighbor_tl_id = self.traffic_light_ids[i]
                # Get full state
                full_state = self._get_intersection_state(neighbor_tl_id)
                # Only include vehicle counts and waiting times (exclude phase info)
                neighbor_state = full_state[:16]
                neighbor_states.append(neighbor_state)
        
        # Concatenate all neighbor states
        if neighbor_states:
            return np.concatenate(neighbor_states)
        else:
            # Return empty array if no neighbors
            return np.array([], dtype=np.float32)
    
    def _get_state(self):
        """
        Get current state of all intersections based on the mode.
        
        Returns:
            dict or np.array: State representation for all agents
        """
        if self.mode == "independent":
            # Return a dictionary with state for each traffic light
            states = {}
            for tl_id in self.traffic_light_ids:
                states[tl_id] = self._get_intersection_state(tl_id)
            return states
            
        elif self.mode == "centralized":
            # Concatenate states from all intersections
            states = []
            for tl_id in self.traffic_light_ids:
                states.append(self._get_intersection_state(tl_id))
            return np.concatenate(states)
            
        elif self.mode == "neighbor_aware":
            # Each intersection gets its own state plus neighbor states
            states = {}
            for tl_id in self.traffic_light_ids:
                own_state = self._get_intersection_state(tl_id)
                neighbor_states = self._get_neighbor_states(tl_id)
                
                # Pad neighbor states if needed
                max_neighbors = min(4 * self.communication_range, self.num_traffic_lights - 1)
                expected_neighbor_size = 16 * max_neighbors
                
                if len(neighbor_states) < expected_neighbor_size:
                    padding = np.zeros(expected_neighbor_size - len(neighbor_states))
                    neighbor_states = np.concatenate([neighbor_states, padding])
                
                # Combine own state with neighbor states
                combined_state = np.concatenate([own_state, neighbor_states])
                states[tl_id] = combined_state
                
            return states
    
    def _get_global_reward(self):
        """
        Calculate global reward based on overall traffic efficiency.
        
        Returns:
            float: Global reward value
        """
        # Get all vehicles in the simulation
        vehicle_ids = traci.vehicle.getIDList()
        
        if not vehicle_ids:
            return 0.0
        
        # Calculate total waiting time across all vehicles
        total_waiting_time = sum(traci.vehicle.getWaitingTime(vid) for vid in vehicle_ids)
        
        # Track emergency vehicles separately (higher priority)
        emergency_vehicles = [vid for vid in vehicle_ids 
                             if traci.vehicle.getVehicleClass(vid) == "emergency"]
        
        emergency_waiting_time = sum(traci.vehicle.getWaitingTime(vid) for vid in emergency_vehicles)
        
        # Update global metrics
        self.global_metrics["total_waiting_time"] = total_waiting_time
        self.global_metrics["emergency_vehicle_delay"] = emergency_waiting_time
        
        # Calculate maximum waiting time (worst case)
        if vehicle_ids:
            max_waiting = max(traci.vehicle.getWaitingTime(vid) for vid in vehicle_ids)
            self.global_metrics["max_waiting_time"] = max_waiting
        
        # Global reward is negative waiting time with extra penalty for emergency vehicles
        reward = -total_waiting_time - (5 * emergency_waiting_time)
        
        return reward
    
    def _get_local_rewards(self):
        """
        Calculate individual rewards for each intersection.
        
        Returns:
            dict: Rewards for each traffic light
        """
        rewards = {}
        
        for tl_id in self.traffic_light_ids:
            # Get incoming lanes for this traffic light
            incoming_lanes = []
            try:
                controlled_links = traci.trafficlight.getControlledLinks(tl_id)
                for link_group in controlled_links:
                    for link in link_group:
                        if link:  # Check if link exists
                            incoming_lane = link[0]
                            incoming_lanes.append(incoming_lane)
            except:
                # If we can't get controlled links, use empty list
                pass
            
            # Get vehicles in the incoming lanes
            vehicle_ids = []
            for lane in incoming_lanes:
                try:
                    vehicle_ids.extend(traci.lane.getLastStepVehicleIDs(lane))
                except:
                    pass
            
            # Calculate waiting time for vehicles in this intersection
            if vehicle_ids:
                local_waiting_time = sum(traci.vehicle.getWaitingTime(vid) for vid in vehicle_ids)
                # Store in metrics
                self.intersection_metrics[tl_id]["waiting_time"] = local_waiting_time
                self.intersection_metrics[tl_id]["vehicle_count"] = len(vehicle_ids)
                
                # Local reward is negative waiting time
                rewards[tl_id] = -local_waiting_time
            else:
                # No vehicles, neutral reward
                self.intersection_metrics[tl_id]["waiting_time"] = 0
                self.intersection_metrics[tl_id]["vehicle_count"] = 0
                rewards[tl_id] = 0
        
        return rewards
    
    def _apply_action(self, tl_id, action):
        """
        Apply traffic light action for a specific intersection.
        
        Args:
            tl_id: Traffic light ID
            action: Action to take (0-3 for phases)
        """
        try:
            # Get current number of phases for this traffic light
            program = traci.trafficlight.getAllProgramLogics(tl_id)[0]
            num_phases = len(program.phases)
            
            # Map action to valid phase (handle case where action > num_phases)
            target_phase = action % num_phases
            
            # Set traffic light phase
            traci.trafficlight.setPhase(tl_id, target_phase)
            
        except Exception as e:
            print(f"Warning: Could not apply action to traffic light {tl_id}: {e}")
            # If setting the phase fails, log the error but continue execution
    
    def reset(self, **kwargs):
        """
        Reset the environment for a new episode.
        
        Returns:
            tuple: (observation, info)
        """
        # Start new SUMO simulation
        self._start_sumo()
        
        # Reset episode tracking
        self.current_step = 0
        self.global_metrics = {
            "total_waiting_time": 0,
            "max_waiting_time": 0,
            "vehicles_processed": 0,
            "emergency_vehicle_delay": 0,
        }
        self.intersection_metrics = {tl_id: {} for tl_id in self.traffic_light_ids}
        
        # Get initial state
        initial_state = self._get_state()
        
        # Initial info dictionary
        info = {
            "traffic_lights": self.traffic_light_ids,
            "num_intersections": self.num_traffic_lights,
            "grid_size": self.grid_size,
            "control_mode": self.mode
        }
        
        return initial_state, info
    
    def step(self, action):
        """
        Execute one environment step.
        
        Args:
            action: Action to take (format depends on mode)
                - If mode="independent": dict mapping tl_id to action
                - If mode="centralized": array of actions for all traffic lights
                - If mode="neighbor_aware": dict mapping tl_id to action
            
        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        try:
            # Apply actions based on mode
            if self.mode == "independent" or self.mode == "neighbor_aware":
                # Action is a dictionary mapping tl_id to action
                for tl_id, tl_action in action.items():
                    self._apply_action(tl_id, tl_action)
                    
            elif self.mode == "centralized":
                # Action is an array with one action per traffic light
                for i, tl_id in enumerate(self.traffic_light_ids):
                    if i < len(action):
                        self._apply_action(tl_id, action[i])
            
            # Run simulation for step_duration seconds
            for _ in range(self.step_duration):
                traci.simulationStep()
            
            # Get new state and reward
            new_state = self._get_state()
            
            # Calculate rewards
            global_reward = self._get_global_reward()
            local_rewards = self._get_local_rewards()
            
            # Combine rewards based on mode
            if self.mode == "centralized":
                reward = global_reward
            else:
                # For independent and neighbor_aware, use local rewards
                reward = local_rewards
            
            # Update step counter
            self.current_step += self.step_duration
            
            # Check if episode is done
            terminated = self.current_step >= self.episode_length
            truncated = False
            
            # Build comprehensive info dictionary
            info = {
                "step": self.current_step,
                "global_metrics": self.global_metrics,
                "intersection_metrics": self.intersection_metrics,
                "global_reward": global_reward,
                "local_rewards": local_rewards
            }
            
            return new_state, reward, terminated, truncated, info
            
        except Exception as e:
            print(f"❌ Error during multi-agent simulation step: {e}")
            
            # Try to recover
            try:
                self._close_sumo()
                self._start_sumo()
                
                # Return safe default state
                if self.mode == "centralized":
                    safe_state = np.zeros(20 * self.num_traffic_lights, dtype=np.float32)
                    safe_reward = -1000
                else:
                    safe_state = {tl_id: np.zeros(20, dtype=np.float32) 
                                 for tl_id in self.traffic_light_ids}
                    safe_reward = {tl_id: -1000 for tl_id in self.traffic_light_ids}
                
                safe_info = {"error": str(e)}
                return safe_state, safe_reward, True, False, safe_info
                
            except Exception as recovery_error:
                print(f"❌ Recovery failed: {recovery_error}")
                
                # Return safe default and terminate
                if self.mode == "centralized":
                    safe_state = np.zeros(20 * self.num_traffic_lights, dtype=np.float32)
                    safe_reward = -1000
                else:
                    safe_state = {tl_id: np.zeros(20, dtype=np.float32) 
                                 for tl_id in self.traffic_light_ids}
                    safe_reward = {tl_id: -1000 for tl_id in self.traffic_light_ids}
                
                safe_info = {
                    "error": str(e),
                    "recovery_error": str(recovery_error)
                }
                return safe_state, safe_reward, True, False, safe_info
    
    def close(self):
        """Close the environment."""
        self._close_sumo()
    
    def render(self, mode='human'):
        """Render the environment (SUMO GUI)."""
        # SUMO handles its own rendering
        pass


def test_multi_environment():
    """Test function to demonstrate multi-intersection environment capabilities."""
    print("Testing Multi-Intersection SUMO Environment")
    print("=" * 50)
    
    # Test with grid network
    env = MultiSumoTrafficEnv(
        sumo_config_file="sumo_scenarios/grid_4x4_tl/grid_4x4.sumocfg",
        grid_size=(4, 4),
        episode_length=60,  # 5 minutes (60 steps x 5 seconds)
        mode="independent"
    )
    
    print(f"Testing basic functionality with {env.num_traffic_lights} traffic lights...")
    
    # Reset environment
    obs, info = env.reset()
    print(f"Environment reset successful")
    print(f"   Observation type: {type(obs)}")
    print(f"   Number of traffic lights: {len(obs) if isinstance(obs, dict) else 'N/A'}")
    
    # Run a few steps with random actions
    for step in range(5):
        # Create random actions based on mode
        if env.mode == "centralized":
            action = np.random.randint(0, 4, size=env.num_traffic_lights)
        else:
            action = {tl_id: np.random.randint(0, 4) for tl_id in env.traffic_light_ids}
        
        # Take step
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"Step {step + 1}:")
        print(f"   Global waiting time: {info['global_metrics']['total_waiting_time']:.2f}s")
        print(f"   Global reward: {info['global_reward']:.2f}")
        
        if terminated or truncated:
            break
    
    env.close()
    print("Multi-intersection environment test completed!")


if __name__ == "__main__":
    test_multi_environment()
