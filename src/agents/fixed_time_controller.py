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