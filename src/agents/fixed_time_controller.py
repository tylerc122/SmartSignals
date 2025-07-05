"""
Fixed-Time Traffic Light Controller

This is a baseline controller that mimics traditional traffic lights.
It follows a rigid, pre-programmed timing pattern regardless of actual traffic conditions.
This serves as our "dumb" baseline to compare against the intelligent RL agent.
"""

import numpy as np
from typing import Any, Dict


class FixedTimeController:
    """
    A simple fixed-time traffic light controller.
    
    This controller cycles through traffic light phases on a fixed timer,
    just like traditional traffic lights in the real world.
    
    Args:
        phase_durations: List of durations (in steps) for each phase
        step_duration: Duration of each simulation step in seconds
    """
    
    def __init__(self, phase_durations: list = None, step_duration: int = 5):
        """
        Initialize the fixed-time controller.
        
        Default timing: 30 seconds per phase (6 steps × 5 seconds each)
        This creates a classic 2-minute cycle (30s NS + 30s EW + 30s NS + 30s EW)
        """
        if phase_durations is None:
            # Default: 30 seconds per phase (6 simulation steps of 5 seconds each)
            self.phase_durations = [6, 6, 6, 6]  # 4 phases, 6 steps each
        else:
            self.phase_durations = phase_durations
        self.step_duration = step_duration
        self.current_phase = 0
        self.steps_in_current_phase = 0
        self.total_steps = 0
        
        # Calculate total cycle time for reporting
        total_cycle_steps = sum(self.phase_durations)
        self.cycle_time = total_cycle_steps * step_duration
        
        print(f"Fixed-Time Controller initialized:")
        print(f"   Phase durations: {self.phase_durations} steps")
        print(f"   Step duration: {step_duration} seconds")
        print(f"   Total cycle time: {self.cycle_time} seconds")
    
    def get_action(self, observation: np.ndarray) -> int:
        """
        Get the next action based on the fixed timing pattern.
        
        Args:
            observation: Current state observation (ignored by fixed-time controller)
            
        Returns:
            action: The traffic light phase to activate (0, 1, 2, or 3)
        """
        # Check if we need to switch to the next phase
        if self.steps_in_current_phase >= self.phase_durations[self.current_phase]:
            # Move to next phase
            self.current_phase = (self.current_phase + 1) % len(self.phase_durations)
            self.steps_in_current_phase = 0
        
        # Record this step
        current_action = self.current_phase
        self.steps_in_current_phase += 1
        self.total_steps += 1
        
        return current_action
    
    def reset(self):
        """Reset the controller to initial state (used between episodes)."""
        self.current_phase = 0
        self.steps_in_current_phase = 0
        self.total_steps = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get controller statistics for analysis."""
        return {
            "controller_type": "Fixed-Time",
            "total_steps": self.total_steps,
            "current_phase": self.current_phase,
            "steps_in_current_phase": self.steps_in_current_phase,
            "cycle_time": self.cycle_time,
            "phase_durations": self.phase_durations
        }
    
    def __str__(self):
        return f"FixedTimeController(cycle={self.cycle_time}s, phase={self.current_phase})"


class AdaptiveFixedTimeController(FixedTimeController):
    """
    A slightly smarter fixed-time controller with different timing patterns.
    
    This variant allows for asymmetric timing (e.g., longer green for busy directions)
    but still doesn't respond to real-time traffic conditions.
    """
    
    def __init__(self, north_south_duration: int = 8, east_west_duration: int = 4, step_duration: int = 5):
        """
        Initialize with different durations for different directions.
        
        Args:
            north_south_duration: Steps for North-South phases
            east_west_duration: Steps for East-West phases  
            step_duration: Duration of each simulation step
        """
        # Assuming phases 0,2 are North-South and phases 1,3 are East-West
        phase_durations = [north_south_duration, east_west_duration, 
                          north_south_duration, east_west_duration]
        
        super().__init__(phase_durations, step_duration)
        
        print(f"Adaptive Fixed-Time Controller:")
        print(f"   North-South phases: {north_south_duration * step_duration} seconds")
        print(f"   East-West phases: {east_west_duration * step_duration} seconds")


class ActuatedController:
    """
    Vehicle-Actuated Traffic Light Controller
    
    This controller mimics real-world actuated traffic signals that respond to
    vehicle presence and demand. It uses vehicle detection data to make intelligent
    timing decisions, representing current industry-standard traffic control technology.
    
    Key Features:
    - Responds to real-time vehicle presence
    - Extends green time when vehicles are detected
    - Skips phases with no vehicle demand
    - Implements minimum and maximum green times
    - Prioritizes high-demand approaches
    """
    
    def __init__(self, 
                 min_green_time: int = 2,  # Minimum green time (steps)
                 max_green_time: int = 10,  # Maximum green time (steps)
                 extension_time: int = 2,   # Extension per vehicle detection (steps)
                 detection_threshold: int = 1,  # Minimum vehicles to trigger phase
                 step_duration: int = 5):
        """
        Initialize the actuated controller.
        
        Args:
            min_green_time: Minimum green time in steps (safety requirement)
            max_green_time: Maximum green time in steps (prevents one direction monopolizing)
            extension_time: Additional time granted per vehicle detection
            detection_threshold: Minimum vehicles needed to call a phase
            step_duration: Duration of each simulation step in seconds
        """
        self.min_green_time = min_green_time
        self.max_green_time = max_green_time
        self.extension_time = extension_time
        self.detection_threshold = detection_threshold
        self.step_duration = step_duration
        
        # State tracking
        self.current_phase = 0
        self.steps_in_current_phase = 0
        self.total_steps = 0
        self.phase_requests = {0: False, 1: False, 2: False, 3: False}
        
        # Lane mappings for 4-phase intersection
        # Phase 0: North-South green (lanes 0,2 incoming)
        # Phase 1: All red (yellow transition)
        # Phase 2: East-West green (lanes 1,3 incoming)
        # Phase 3: All red (yellow transition)
        self.phase_to_lanes = {
            0: [0, 2],  # North-South lanes (n2c, s2c)
            1: [],      # Yellow transition
            2: [1, 3],  # East-West lanes (e2c, w2c)
            3: []       # Yellow transition
        }
        
        # Green phases only (skip yellow phases for demand detection)
        self.green_phases = [0, 2]
        
        print(f"Actuated Controller initialized:")
        print(f"   Min green time: {min_green_time * step_duration} seconds")
        print(f"   Max green time: {max_green_time * step_duration} seconds")
        print(f"   Extension time: {extension_time * step_duration} seconds")
        print(f"   Detection threshold: {detection_threshold} vehicles")
    
    def _parse_observation(self, observation: np.ndarray) -> Dict[str, Any]:
        """
        Parse the observation vector into meaningful traffic data.
        
        Args:
            observation: 20-dimensional state vector
            
        Returns:
            dict: Parsed traffic data
        """
        # Extract components from observation
        vehicle_counts = observation[:8]      # Vehicles waiting per lane
        waiting_times = observation[8:16]     # Average waiting time per lane
        phase_one_hot = observation[16:20]    # Current phase (one-hot)
        
        # Focus on incoming lanes (first 4 values)
        incoming_vehicles = vehicle_counts[:4]  # [north, east, south, west]
        incoming_waiting = waiting_times[:4]
        
        # Current phase
        current_phase = int(np.argmax(phase_one_hot))
        
        return {
            'incoming_vehicles': incoming_vehicles,
            'incoming_waiting': incoming_waiting,
            'current_phase': current_phase,
            'total_vehicles': int(np.sum(incoming_vehicles)),
            'max_waiting_time': float(np.max(incoming_waiting))
        }
    
    def _detect_phase_demands(self, traffic_data: Dict[str, Any]) -> None:
        """
        Detect which phases have vehicle demand (like real inductive loops).
        
        Args:
            traffic_data: Parsed traffic observation data
        """
        incoming_vehicles = traffic_data['incoming_vehicles']
        
        # Clear previous requests
        self.phase_requests = {0: False, 1: False, 2: False, 3: False}
        
        # Check demand for each green phase
        for phase in self.green_phases:
            lanes = self.phase_to_lanes[phase]
            total_demand = sum(incoming_vehicles[lane] for lane in lanes)
            
            # Phase has demand if vehicles >= threshold
            if total_demand >= self.detection_threshold:
                self.phase_requests[phase] = True
                # Also request the preceding yellow phase if needed
                if phase > 0:
                    self.phase_requests[phase - 1] = True
    
    def _should_extend_green(self, traffic_data: Dict[str, Any]) -> bool:
        """
        Determine if current green phase should be extended.
        
        Args:
            traffic_data: Parsed traffic observation data
            
        Returns:
            bool: True if phase should be extended
        """
        # Only extend green phases
        if self.current_phase not in self.green_phases:
            return False
        
        # Don't extend beyond maximum
        if self.steps_in_current_phase >= self.max_green_time:
            return False
        
        # Check if current phase still has vehicles
        current_lanes = self.phase_to_lanes[self.current_phase]
        current_demand = sum(traffic_data['incoming_vehicles'][lane] for lane in current_lanes)
        
        # Extend if vehicles are still present
        return current_demand >= self.detection_threshold
    
    def _should_switch_phase(self, traffic_data: Dict[str, Any]) -> bool:
        """
        Determine if we should switch to a different phase.
        
        Args:
            traffic_data: Parsed traffic observation data
            
        Returns:
            bool: True if should switch phases
        """
        # Must meet minimum green time first
        if self.steps_in_current_phase < self.min_green_time:
            return False
        
        # For green phases, check if we should continue or switch
        if self.current_phase in self.green_phases:
            # Switch if no more vehicles on current phase and others are waiting
            current_lanes = self.phase_to_lanes[self.current_phase]
            current_demand = sum(traffic_data['incoming_vehicles'][lane] for lane in current_lanes)
            
            # Check if other phases have demand
            other_phases_have_demand = any(
                self.phase_requests[phase] 
                for phase in self.green_phases 
                if phase != self.current_phase
            )
            
            # Switch if current phase is empty and others have demand
            if current_demand < self.detection_threshold and other_phases_have_demand:
                return True
            
            # Or if we've reached maximum green time
            if self.steps_in_current_phase >= self.max_green_time:
                return True
        
        # For yellow phases, switch after 1 step (quick transition)
        elif self.current_phase not in self.green_phases:
            return self.steps_in_current_phase >= 1
        
        return False
    
    def _get_next_phase(self, traffic_data: Dict[str, Any]) -> int:
        """
        Determine the next phase to activate based on demand.
        
        Args:
            traffic_data: Parsed traffic observation data
            
        Returns:
            int: Next phase to activate
        """
        # If currently in yellow, determine which green phase to go to
        if self.current_phase not in self.green_phases:
            # Check which green phases have demand
            demanding_phases = [p for p in self.green_phases if self.phase_requests[p]]
            
            if demanding_phases:
                # Choose phase with highest demand
                max_demand = 0
                best_phase = demanding_phases[0]
                
                for phase in demanding_phases:
                    lanes = self.phase_to_lanes[phase]
                    demand = sum(traffic_data['incoming_vehicles'][lane] for lane in lanes)
                    if demand > max_demand:
                        max_demand = demand
                        best_phase = phase
                
                return best_phase
            else:
                # No demand, default to next green phase
                return self.green_phases[0]
        
        # If currently in green, go to yellow first (realistic transition)
        else:
            # Find next green phase with demand
            current_green_idx = self.green_phases.index(self.current_phase)
            next_green_idx = (current_green_idx + 1) % len(self.green_phases)
            next_green_phase = self.green_phases[next_green_idx]
            
            # If next phase has demand, go to yellow first
            if self.phase_requests[next_green_phase]:
                return next_green_phase - 1 if next_green_phase > 0 else 1
            else:
                # Skip to the phase after if no demand
                return next_green_phase
    
    def get_action(self, observation: np.ndarray) -> int:
        """
        Get the next action based on actuated logic and real-time traffic data.
        
        Args:
            observation: Current state observation with vehicle counts and waiting times
            
        Returns:
            action: The traffic light phase to activate (0, 1, 2, or 3)
        """
        # Parse observation into traffic data
        traffic_data = self._parse_observation(observation)
        
        # Detect phase demands (like inductive loop detectors)
        self._detect_phase_demands(traffic_data)
        
        # Actuated logic decision tree
        if self._should_switch_phase(traffic_data):
            # Switch to next phase with demand
            self.current_phase = self._get_next_phase(traffic_data)
            self.steps_in_current_phase = 0
        
        # Record this step
        current_action = self.current_phase
        self.steps_in_current_phase += 1
        self.total_steps += 1
        
        return current_action
    
    def reset(self):
        """Reset the controller to initial state (used between episodes)."""
        self.current_phase = 0
        self.steps_in_current_phase = 0
        self.total_steps = 0
        self.phase_requests = {0: False, 1: False, 2: False, 3: False}
    
    def get_stats(self) -> Dict[str, Any]:
        """Get controller statistics for analysis."""
        return {
            "controller_type": "Vehicle-Actuated",
            "total_steps": self.total_steps,
            "current_phase": self.current_phase,
            "steps_in_current_phase": self.steps_in_current_phase,
            "min_green_time": self.min_green_time * self.step_duration,
            "max_green_time": self.max_green_time * self.step_duration,
            "detection_threshold": self.detection_threshold
        }
    
    def __str__(self):
        return f"ActuatedController(min={self.min_green_time}s, max={self.max_green_time}s, phase={self.current_phase})" 