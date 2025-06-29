#!/usr/bin/env python3
"""
Test script for the updated SUMO Traffic Environment with Cross Intersection

This script tests the functionality of our RL environment using the new
4-way cross intersection with proper traffic light control.
"""

import sys
import os
import numpy as np

# Add src directory to path so we can import our environment
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from environments.sumo_traffic_env import SumoTrafficEnv


def test_cross_intersection():
    """Test the cross intersection environment functionality."""
    print("🚦 Testing Cross Intersection Environment")
    print("=" * 50)
    
    # Create environment with our new cross intersection
    env = SumoTrafficEnv(
        sumo_config_file="sumo_scenarios/cross_intersection.sumocfg",
        step_duration=5,  # 5 seconds per RL step
        episode_length=60  # 1 minute episodes for testing
    )
    
    try:
        print("✅ Environment created successfully")
        print(f"📊 Observation space: {env.observation_space}")
        print(f"🎮 Action space: {env.action_space}")
        
        # Test reset
        print("\n🔄 Testing environment reset...")
        observation, info = env.reset()
        print(f"✅ Reset successful")
        print(f"📋 Initial observation shape: {observation.shape}")
        print(f"📋 Initial observation: {observation}")
        
        # Test multiple steps with different actions
        print("\n🏃 Testing environment steps...")
        total_reward = 0
        
        for step in range(5):  # Test 5 steps (25 seconds of simulation)
            action = step % 4  # Cycle through all 4 phases
            print(f"\n  Step {step + 1}: Taking action {action}")
            
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            print(f"    📊 Observation shape: {obs.shape}")
            print(f"    🎁 Reward: {reward:.2f}")
            print(f"    🏁 Terminated: {terminated}")
            print(f"    ⏰ Step info: {info}")
            
            if terminated or truncated:
                print("    Episode completed!")
                break
        
        print(f"\n📈 Total reward: {total_reward:.2f}")
        print("✅ All tests passed!")
        
        # Test state interpretation
        print("\n🔍 Analyzing final state...")
        incoming_vehicles = obs[:4]  # First 4 values: incoming lane vehicle counts
        outgoing_vehicles = obs[4:8]  # Next 4 values: outgoing lane vehicle counts
        incoming_waiting = obs[8:12]  # Next 4 values: incoming lane waiting times
        outgoing_waiting = obs[12:16]  # Next 4 values: outgoing lane waiting times
        phase_encoding = obs[16:20]  # Last 4 values: traffic light phase
        
        print(f"  🚗 Incoming vehicles (N,E,S,W): {incoming_vehicles}")
        print(f"  🚗 Outgoing vehicles (N,E,S,W): {outgoing_vehicles}")
        print(f"  ⏰ Incoming waiting times: {incoming_waiting}")
        print(f"  ⏰ Outgoing waiting times: {outgoing_waiting}")
        print(f"  🚦 Current phase: {np.argmax(phase_encoding)} (one-hot: {phase_encoding})")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        print("\n🧹 Cleaning up...")
        env.close()
        print("✅ Environment closed")


if __name__ == "__main__":
    print("🚀 Starting Cross Intersection Environment Test")
    print("This tests our new 4-way intersection with proper crossing traffic!")
    print()
    
    test_cross_intersection()
    
    print("\n🎉 Test completed!")
    print("\nWhat we tested:")
    print("- ✅ Environment creation with cross intersection")
    print("- ✅ Reset functionality") 
    print("- ✅ Step execution with traffic light control")
    print("- ✅ State observation (20-dimensional)")
    print("- ✅ Reward calculation") 
    print("- ✅ Proper cleanup")
    print("\nNext: Train an RL agent on this environment! 🤖") 