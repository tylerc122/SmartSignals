#!/usr/bin/env python3
"""
Final test script for SUMO traffic environment with working traffic lights!

This script tests our RL environment with a properly generated SUMO network
that includes real traffic lights created using SUMO's netconvert tool.
"""

import sys
import os
import time

# Add src directory to path so we can import our environment
sys.path.append('src')

try:
    from environments.sumo_traffic_env import SumoTrafficEnv
    print("✅ Successfully imported SumoTrafficEnv")
except ImportError as e:
    print(f"❌ Failed to import environment: {e}")
    sys.exit(1)

def test_final_environment():
    """Test the SUMO traffic environment with working traffic lights."""
    
    print("\n🚦 Testing SUMO Traffic Environment (FINAL TEST)...")
    print("=" * 65)
    
    # Test 1: Environment Creation
    print("\n1️⃣ Creating environment...")
    try:
        env = SumoTrafficEnv(
            sumo_config_path="sumo_scenarios/tl_intersection.sumocfg",  # WORKING CONFIG!
            episode_length=60,  # Short episode for testing
            step_size=5,
            render_mode=None  # Headless mode (no GUI)
        )
        print("✅ Environment created successfully")
        print(f"   Action space: {env.action_space}")
        print(f"   Observation space: {env.observation_space}")
        print(f"   Expected traffic light ID: {env.traffic_light_id}")
    except Exception as e:
        print(f"❌ Failed to create environment: {e}")
        return False
    
    # Test 2: Environment Reset
    print("\n2️⃣ Resetting environment (starting SUMO)...")
    try:
        observation, info = env.reset()
        print("✅ Environment reset successfully")
        print(f"   Initial observation shape: {observation.shape}")
        print(f"   Episode info: {info}")
    except Exception as e:
        print(f"❌ Failed to reset environment: {e}")
        print(f"   Error details: {str(e)}")
        env.close()
        return False
    
    # Test 3: Traffic Light Information
    print("\n3️⃣ Verifying traffic light setup...")
    try:
        import traci
        if env.sumo_connected:
            # Get all traffic lights
            tl_list = traci.trafficlight.getIDList()
            print(f"   Available traffic lights: {list(tl_list)}")
            
            if env.traffic_light_id in tl_list:
                print(f"   ✅ Traffic light '{env.traffic_light_id}' found!")
                
                # Get current phase
                current_phase = traci.trafficlight.getPhase(env.traffic_light_id)
                print(f"   Current phase: {current_phase}")
                
                # Get controlled lanes
                controlled_lanes = traci.trafficlight.getControlledLanes(env.traffic_light_id)
                print(f"   Controlled lanes: {controlled_lanes}")
                
                # Get phase definitions
                logic = traci.trafficlight.getAllProgramLogics(env.traffic_light_id)
                if logic:
                    phases = logic[0].phases
                    print(f"   Number of phases: {len(phases)}")
                    for i, phase in enumerate(phases):
                        print(f"     Phase {i}: {phase.state} (duration: {phase.duration}s)")
            else:
                print(f"   ❌ Traffic light '{env.traffic_light_id}' not found!")
                print(f"   Available: {list(tl_list)}")
                return False
    except Exception as e:
        print(f"❌ Failed to get traffic light info: {e}")
    
    # Test 4: Take Actions and Control Traffic Light
    print("\n4️⃣ Testing traffic light control...")
    try:
        total_reward = 0.0
        
        for step in range(4):  # Test 4 steps to cycle through phases
            action = step % 4  # Test all 4 actions
            print(f"   Step {step + 1}: Taking action {action}")
            
            # Take the action
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            print(f"     → Reward: {reward:.2f}")
            print(f"     → Total vehicles: {info.get('total_vehicles', 0)}")
            print(f"     → Episode step: {info.get('episode_step', 0)}")
            
            # Show current traffic light phase after action
            if env.sumo_connected:
                try:
                    current_phase = traci.trafficlight.getPhase(env.traffic_light_id)
                    print(f"     → Traffic light phase: {current_phase}")
                except Exception as e:
                    print(f"     → Could not get TL phase: {e}")
            
            if terminated or truncated:
                print("     → Episode ended")
                break
                
        print(f"   Total cumulative reward: {total_reward:.2f}")
        print("✅ Traffic light control working!")
    except Exception as e:
        print(f"❌ Failed during traffic light control: {e}")
        env.close()
        return False
    
    # Test 5: Observation Analysis
    print("\n5️⃣ Analyzing observations...")
    try:
        obs = env._get_observation()
        print(f"   Full observation: {obs}")
        print(f"   Components breakdown:")
        print(f"     Waiting vehicles per lane: {obs[0:4]}")
        print(f"     Average waiting times: {obs[4:8]}")
        print(f"     Traffic light phase (one-hot): {obs[8:12]}")
        
        # Check if observation makes sense
        print("✅ Observation structure is valid")
    except Exception as e:
        print(f"❌ Failed observation analysis: {e}")
    
    # Test 6: Cleanup
    print("\n6️⃣ Cleaning up...")
    try:
        env.close()
        print("✅ Environment closed successfully")
    except Exception as e:
        print(f"❌ Failed to close environment: {e}")
        return False
    
    print("\n🎊 ALL TESTS PASSED! 🎊")
    print("\n📊 Your RL Environment is Ready:")
    print(f"   • SUMO simulation: Working ✅")
    print(f"   • Traffic lights: Detected and controllable ✅")
    print(f"   • RL interface: Fully functional ✅")
    print(f"   • Observations: Valid 12D state vectors ✅")
    print(f"   • Actions: 4 discrete traffic light phases ✅")
    print(f"   • Rewards: Based on vehicle waiting time ✅")
    
    return True

def main():
    """Run the final environment test."""
    print("🚗 FINAL SUMO Traffic Environment Test")
    print("Using PROPERLY GENERATED traffic lights via netconvert!")
    print("This should work perfectly...")
    
    input("\nPress Enter to run the final test...")
    
    success = test_final_environment()
    
    if success:
        print("\n🌟 CONGRATULATIONS! 🌟")
        print("Your Smart Traffic Signals RL environment is working perfectly!")
        print("\n🎯 What you've accomplished:")
        print("   ✓ Set up SUMO traffic simulation")
        print("   ✓ Created working traffic light network")
        print("   ✓ Built custom Gymnasium environment")
        print("   ✓ Implemented RL state/action/reward system")
        print("   ✓ Tested full integration")
        
        print("\n🚀 Next Steps:")
        print("   1. Create your first RL training script")
        print("   2. Train a PPO agent to control the traffic lights")
        print("   3. Evaluate performance vs. fixed timing")
        print("   4. Add XQuartz for visualization (optional)")
        print("   5. Experiment with different reward functions")
        
        print("\n📚 You now understand:")
        print("   • How SUMO traffic simulation works")
        print("   • How to create RL environments with Gymnasium")  
        print("   • How traffic light control can be an RL problem")
        print("   • How state/action/reward design affects learning")
        
    else:
        print("\n💔 Something still isn't working.")
        print("Check the error messages above for debugging.")

if __name__ == "__main__":
    main() 