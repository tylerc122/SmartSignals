"""
Traffic Scenario Generator for Phase 2!!

traffic_scenario_generator.py generates diverse traffic demand patterns to test the robustness
of our RL agent across realistic traffic conditions. Instead of using a single
fixed pattern, it creates 100+ (for now) varied scenarios to validate performance.

Essentially, we take 11 prebuilt traffic patterns with their own intrinsic qualities, i.e one has heavy
north-south traffic, light east-west traffic and a different one has even distribution of traffic. We then
apply random multipliers to each direction (+- 15% variation) to create a new scenario. Freaking awesome.
"""

import os
import random
import numpy as np
from typing import Dict, List, Tuple, Optional
import xml.etree.ElementTree as ET
from datetime import datetime
import json


class TrafficScenarioGenerator:
    """
    Generates diverse traffic scenarios for comprehensive RL agent validation.
    
    Creates SUMO route files with varying demand patterns including:
    - Rush hour vs off-peak patterns
    - Directional imbalances (heavy north-south, light east-west)
    - Emergency scenarios with sudden spikes
    - Gradual buildup and decay patterns
    """
    
    def __init__(self, base_scenario_dir: str = "sumo_scenarios"):
        """
        Initialize the traffic scenario generator.
        
        Args:
            base_scenario_dir: Directory containing base SUMO files
        """
        self.base_scenario_dir = base_scenario_dir
        self.scenario_patterns = self._define_scenario_patterns()
        
        # Keep track of generated scenarios
        self.generated_scenarios = []
    
    def _define_scenario_patterns(self) -> Dict:
        """
        Define different traffic demand patterns for scenario generation.
        
        Returns:
            Dictionary of scenario patterns with demand multipliers
        """
        return {
            # Balanced patterns
            "balanced_light": {
                "description": "Light traffic, balanced in all directions",
                "base_demand": 100,
                "direction_multipliers": {"n2s": 1.0, "s2n": 1.0, "e2w": 1.0, "w2e": 1.0},
                "turning_ratios": {"through": 0.6, "right": 0.25, "left": 0.15}
            },
            
            "balanced_moderate": {
                "description": "Moderate traffic, balanced in all directions", 
                "base_demand": 200,
                "direction_multipliers": {"n2s": 1.0, "s2n": 1.0, "e2w": 1.0, "w2e": 1.0},
                "turning_ratios": {"through": 0.6, "right": 0.25, "left": 0.15}
            },
            
            "balanced_heavy": {
                "description": "Heavy traffic, balanced in all directions",
                "base_demand": 400,
                "direction_multipliers": {"n2s": 1.0, "s2n": 1.0, "e2w": 1.0, "w2e": 1.0},
                "turning_ratios": {"through": 0.6, "right": 0.25, "left": 0.15}
            },
            
            # Rush hour patterns
            "rush_hour_ns": {
                "description": "Rush hour with heavy north-south flow",
                "base_demand": 300,
                "direction_multipliers": {"n2s": 1.5, "s2n": 1.4, "e2w": 0.6, "w2e": 0.7},
                "turning_ratios": {"through": 0.7, "right": 0.2, "left": 0.1}
            },
            
            "rush_hour_ew": {
                "description": "Rush hour with heavy east-west flow",
                "base_demand": 300,
                "direction_multipliers": {"n2s": 0.7, "s2n": 0.6, "e2w": 1.4, "w2e": 1.5},
                "turning_ratios": {"through": 0.7, "right": 0.2, "left": 0.1}
            },
            
            # Unbalanced patterns
            "unbalanced_north_heavy": {
                "description": "Heavy northbound traffic, light other directions",
                "base_demand": 250,
                "direction_multipliers": {"n2s": 0.3, "s2n": 1.8, "e2w": 0.8, "w2e": 0.9},
                "turning_ratios": {"through": 0.6, "right": 0.25, "left": 0.15}
            },
            
            "unbalanced_east_heavy": {
                "description": "Heavy eastbound traffic, light other directions",
                "base_demand": 250,
                "direction_multipliers": {"n2s": 0.8, "s2n": 0.9, "e2w": 0.3, "w2e": 1.8},
                "turning_ratios": {"through": 0.6, "right": 0.25, "left": 0.15}
            },
            
            # Emergency/event patterns
            "emergency_spike": {
                "description": "Sudden traffic spike in one direction (emergency, event)",
                "base_demand": 150,
                "direction_multipliers": {"n2s": 3.0, "s2n": 0.2, "e2w": 0.5, "w2e": 0.3},
                "turning_ratios": {"through": 0.8, "right": 0.15, "left": 0.05}
            },
            
            "event_dispersal": {
                "description": "Event ending - heavy outbound from all directions",
                "base_demand": 350,
                "direction_multipliers": {"n2s": 1.2, "s2n": 1.3, "e2w": 1.1, "w2e": 1.4},
                "turning_ratios": {"through": 0.5, "right": 0.3, "left": 0.2}
            },
            
            # Off-peak patterns
            "off_peak_light": {
                "description": "Late night/early morning light traffic",
                "base_demand": 50,
                "direction_multipliers": {"n2s": 0.8, "s2n": 1.2, "e2w": 1.0, "w2e": 0.9},
                "turning_ratios": {"through": 0.5, "right": 0.3, "left": 0.2}
            },
            
            "weekend_moderate": {
                "description": "Weekend moderate traffic with more turns",
                "base_demand": 150,
                "direction_multipliers": {"n2s": 1.0, "s2n": 1.0, "e2w": 1.0, "w2e": 1.0},
                "turning_ratios": {"through": 0.4, "right": 0.35, "left": 0.25}
            }
        }
    
    def generate_scenario(self, pattern_name: str, simulation_duration: int = 300,
                         random_variation: float = 0.1) -> Dict:
        """
        Generate a single traffic scenario based on a pattern.
        
        Args:
            pattern_name: Name of the traffic pattern to use
            simulation_duration: Duration of simulation in seconds
            random_variation: Random variation factor (0.0-1.0)
            
        Returns:
            Dictionary containing scenario metadata and file paths
        """
        if pattern_name not in self.scenario_patterns:
            raise ValueError(f"Unknown pattern: {pattern_name}")
        
        pattern = self.scenario_patterns[pattern_name]
        
        # Apply random variation to make scenarios more diverse
        varied_multipliers = {}
        for direction, multiplier in pattern["direction_multipliers"].items():
            variation = random.uniform(1 - random_variation, 1 + random_variation)
            varied_multipliers[direction] = multiplier * variation
        
        # Generate unique scenario ID
        scenario_id = f"{pattern_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{random.randint(1000, 9999)}"
        
        # Create scenario directory
        scenario_dir = os.path.join("sumo_scenarios", "phase2_scenarios", scenario_id)
        os.makedirs(scenario_dir, exist_ok=True)
        
        # Generate route file
        route_file = os.path.join(scenario_dir, f"{scenario_id}.rou.xml")
        self._generate_route_file(
            route_file, 
            pattern["base_demand"], 
            varied_multipliers,
            pattern["turning_ratios"],
            simulation_duration
        )
        
        # Create SUMO config file
        config_file = os.path.join(scenario_dir, f"{scenario_id}.sumocfg")
        self._generate_config_file(config_file, route_file)
        
        # Store scenario metadata
        scenario_metadata = {
            "scenario_id": scenario_id,
            "pattern_name": pattern_name,
            "description": pattern["description"],
            "base_demand": pattern["base_demand"],
            "direction_multipliers": varied_multipliers,
            "turning_ratios": pattern["turning_ratios"],
            "simulation_duration": simulation_duration,
            "config_file": config_file,
            "route_file": route_file,
            "generated_at": datetime.now().isoformat()
        }
        
        self.generated_scenarios.append(scenario_metadata)
        
        return scenario_metadata
    
    def _generate_route_file(self, route_file: str, base_demand: int, 
                           direction_multipliers: Dict, turning_ratios: Dict,
                           simulation_duration: int):
        """
        Generate a SUMO route file with specified traffic patterns.
        
        Args:
            route_file: Path to output route file
            base_demand: Base demand in vehicles per hour
            direction_multipliers: Multipliers for each direction
            turning_ratios: Ratios for through/right/left turns
            simulation_duration: Duration of simulation
        """
        # Create XML structure
        root = ET.Element("routes")
        root.set("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance")
        root.set("xsi:noNamespaceSchemaLocation", "http://sumo.dlr.de/xsd/routes_file.xsd")
        
        # Vehicle type
        vtype = ET.SubElement(root, "vType")
        vtype.set("id", "car")
        vtype.set("accel", "2.6")
        vtype.set("decel", "4.5")
        vtype.set("sigma", "0.5")
        vtype.set("length", "5.0")
        vtype.set("maxSpeed", "13.89")
        vtype.set("guiShape", "passenger")
        
        # Define all possible routes
        routes = [
            # North routes
            ("north_to_south", "n2c c2s"),
            ("north_to_east", "n2c c2e"),
            ("north_to_west", "n2c c2w"),
            # South routes
            ("south_to_north", "s2c c2n"),
            ("south_to_west", "s2c c2w"),
            ("south_to_east", "s2c c2e"),
            # East routes
            ("east_to_west", "e2c c2w"),
            ("east_to_north", "e2c c2n"),
            ("east_to_south", "e2c c2s"),
            # West routes
            ("west_to_east", "w2c c2e"),
            ("west_to_south", "w2c c2s"),
            ("west_to_north", "w2c c2n")
        ]
        
        # Add route definitions
        for route_id, edges in routes:
            route_elem = ET.SubElement(root, "route")
            route_elem.set("id", route_id)
            route_elem.set("edges", edges)
        
        # Generate traffic flows
        flow_id = 0
        for direction, multiplier in direction_multipliers.items():
            # Calculate demand for this direction
            direction_demand = int(base_demand * multiplier)
            
            # Split demand by turning movements
            through_demand = int(direction_demand * turning_ratios["through"])
            right_demand = int(direction_demand * turning_ratios["right"])
            left_demand = int(direction_demand * turning_ratios["left"])
            
            # Map direction to route prefixes
            route_mapping = {
                "n2s": [("north_to_south", through_demand), ("north_to_east", right_demand), ("north_to_west", left_demand)],
                "s2n": [("south_to_north", through_demand), ("south_to_west", right_demand), ("south_to_east", left_demand)],
                "e2w": [("east_to_west", through_demand), ("east_to_north", right_demand), ("east_to_south", left_demand)],
                "w2e": [("west_to_east", through_demand), ("west_to_south", right_demand), ("west_to_north", left_demand)]
            }
            
            # Add flows for this direction
            for route_name, demand in route_mapping[direction]:
                if demand > 0:  # Only add flows with positive demand
                    flow = ET.SubElement(root, "flow")
                    flow.set("id", f"flow_{flow_id}")
                    flow.set("route", route_name)
                    flow.set("begin", "0")
                    flow.set("end", str(simulation_duration))
                    flow.set("vehsPerHour", str(demand))
                    flow.set("type", "car")
                    flow_id += 1
        
        # Write to file
        tree = ET.ElementTree(root)
        ET.indent(tree, space="    ")
        tree.write(route_file, encoding="utf-8", xml_declaration=True)
    
    def _generate_config_file(self, config_file: str, route_file: str):
        """
        Generate a SUMO configuration file for the scenario.
        
        Args:
            config_file: Path to output config file
            route_file: Path to the route file
        """
        # Create XML structure
        root = ET.Element("configuration")
        root.set("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance")
        root.set("xsi:noNamespaceSchemaLocation", "http://sumo.dlr.de/xsd/sumoConfiguration.xsd")
        
        # Input section
        input_elem = ET.SubElement(root, "input")
        net_file = ET.SubElement(input_elem, "net-file")
        net_file.set("value", "../../cross_intersection.net.xml")  # Relative path to base network
        route_files = ET.SubElement(input_elem, "route-files")
        route_files.set("value", os.path.basename(route_file))
        
        # Time section
        time_elem = ET.SubElement(root, "time")
        begin = ET.SubElement(time_elem, "begin")
        begin.set("value", "0")
        end = ET.SubElement(time_elem, "end")
        end.set("value", "300")
        step_length = ET.SubElement(time_elem, "step-length")
        step_length.set("value", "1")
        
        # Report section
        report_elem = ET.SubElement(root, "report")
        verbose = ET.SubElement(report_elem, "verbose")
        verbose.set("value", "false")
        no_step_log = ET.SubElement(report_elem, "no-step-log")
        no_step_log.set("value", "true")
        duration_log = ET.SubElement(report_elem, "duration-log.statistics")
        duration_log.set("value", "true")
        
        # Write to file
        tree = ET.ElementTree(root)
        ET.indent(tree, space="    ")
        tree.write(config_file, encoding="utf-8", xml_declaration=True)
    
    def generate_scenario_batch(self, num_scenarios: int = 100, 
                               patterns: Optional[List[str]] = None,
                               simulation_duration: int = 300) -> List[Dict]:
        """
        Generate a batch of diverse traffic scenarios for Phase 2 validation.
        
        Args:
            num_scenarios: Number of scenarios to generate
            patterns: List of pattern names to use (if None, uses all patterns)
            simulation_duration: Duration of each scenario
            
        Returns:
            List of scenario metadata dictionaries
        """
        if patterns is None:
            patterns = list(self.scenario_patterns.keys())
        
        generated_scenarios = []
        
        print(f"   Generating {num_scenarios} traffic scenarios for Phase 2...")
        print(f"   Using {len(patterns)} different traffic patterns")
        print(f"   Simulation duration: {simulation_duration} seconds each")
        
        for i in range(num_scenarios):
            # Select pattern (cycle through available patterns)
            pattern_name = patterns[i % len(patterns)]
            
            # Generate scenario with random variation
            scenario = self.generate_scenario(
                pattern_name, 
                simulation_duration, 
                random_variation=0.15  # 15% random variation
            )
            
            generated_scenarios.append(scenario)
            
            # Progress update
            if (i + 1) % 10 == 0:
                print(f"   Generated {i + 1}/{num_scenarios} scenarios...")
        
        print(f"✅ Successfully generated {num_scenarios} scenarios!")
        
        # Save scenario index
        self._save_scenario_index(generated_scenarios)
        
        return generated_scenarios
    
    def _save_scenario_index(self, scenarios: List[Dict]):
        """
        Save an index of all generated scenarios for easy reference.
        
        Args:
            scenarios: List of scenario metadata
        """
        index_file = "sumo_scenarios/phase2_scenarios/scenario_index.json"
        os.makedirs(os.path.dirname(index_file), exist_ok=True)
        
        index_data = {
            "generated_at": datetime.now().isoformat(),
            "total_scenarios": len(scenarios),
            "scenarios": scenarios
        }
        
        with open(index_file, 'w') as f:
            json.dump(index_data, f, indent=2)
        
        print(f"Scenario index saved to: {index_file}")
    
    def get_scenario_summary(self) -> Dict:
        """
        Get a summary of all generated scenarios.
        
        Returns:
            Dictionary with scenario statistics
        """
        if not self.generated_scenarios:
            return {"total_scenarios": 0, "patterns": {}}
        
        pattern_counts = {}
        for scenario in self.generated_scenarios:
            pattern = scenario["pattern_name"]
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
        
        return {
            "total_scenarios": len(self.generated_scenarios),
            "patterns": pattern_counts,
            "scenario_types": list(self.scenario_patterns.keys())
        }


def main():
    """Demo function to show how to use the traffic scenario generator."""
    print("🚦 Traffic Scenario Generator Demo")
    print("=" * 50)
    
    # Initialize generator
    generator = TrafficScenarioGenerator()
    
    # Generate a single scenario
    print("\n1. Generating a single rush hour scenario...")
    scenario = generator.generate_scenario("rush_hour_ns", simulation_duration=300)
    print(f"   Generated: {scenario['scenario_id']}")
    print(f"   Description: {scenario['description']}")
    print(f"   Config file: {scenario['config_file']}")
    
    # Generate a small batch
    print("\n2. Generating a batch of 10 scenarios...")
    batch = generator.generate_scenario_batch(num_scenarios=10)
    
    # Show summary
    print("\n3. Scenario Summary:")
    summary = generator.get_scenario_summary()
    print(f"   Total scenarios: {summary['total_scenarios']}")
    print(f"   Pattern distribution: {summary['patterns']}")
    
    print("\n✅ Demo completed!")


if __name__ == "__main__":
    main() 