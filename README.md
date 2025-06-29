# Smart Traffic Signals - Reinforcement Learning Project

A reinforcement learning system that trains AI agents to control traffic lights for optimal traffic flow using SUMO traffic simulation.

## 🚀 Quick Start

1. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

2. **Test the environment:**
   ```bash
   python test_environment.py
   ```

## 📁 Project Structure

```
Smart_Signals/
├── src/
│   └── environments/
│       ├── __init__.py
│       └── sumo_traffic_env.py     # Main RL environment
├── sumo_scenarios/
│   ├── tl_intersection.net.xml     # Traffic network with lights
│   ├── tl_intersection.sumocfg     # SUMO configuration
│   └── simple_routes.rou.xml       # Vehicle routes
├── configs/
│   └── training_config.yaml        # RL training parameters
├── models/                         # (Empty - for trained models)
├── results/                        # (Empty - for training results)
├── test_environment.py             # Environment test script
└── requirements.txt                # Python dependencies
```

## 🧠 How It Works

### Environment (`SumoTrafficEnv`)

- **State Space**: 12-dimensional vector containing:

  - Waiting vehicles per lane (4 values)
  - Average waiting times per lane (4 values)
  - Current traffic light phase (4 values, one-hot encoded)

- **Action Space**: 4 discrete actions (traffic light phases)

  - Actions automatically map to available phases in the traffic light

- **Reward Function**: Negative total waiting time of all vehicles
  - Encourages the agent to minimize traffic delays

### Traffic Simulation

- Uses SUMO (Simulation of Urban MObility) for realistic traffic simulation
- Single intersection with traffic light "B1"
- Continuous vehicle flows from all directions
- Realistic vehicle dynamics and routing

## 🎯 Current Status

✅ **Working Components:**

- SUMO traffic simulation
- Traffic light detection and control
- RL environment (Gymnasium-compatible)
- State observation system
- Action mapping system
- Reward calculation
- Full integration testing

## 🚀 Next Steps

1. **Create training script** using Stable Baselines3
2. **Train RL agent** (PPO recommended)
3. **Evaluate performance** vs fixed-time signals
4. **Add visualization** (requires XQuartz on macOS)
5. **Experiment with different reward functions**

## 🔧 Technical Details

- **RL Framework**: Gymnasium (OpenAI Gym successor)
- **Traffic Simulation**: SUMO 1.23.1
- **Recommended RL Library**: Stable Baselines3
- **Neural Networks**: PyTorch backend
- **Communication**: TraCI (Traffic Control Interface)

## 🎮 Testing

Run the test script to verify everything works:

```bash
python test_environment.py
```

This will test:

- Environment creation
- SUMO connection
- Traffic light control
- Observation collection
- Reward calculation

## 📚 Learning Resources

- [SUMO Documentation](https://sumo.dlr.de/docs/)
- [Stable Baselines3](https://stable-baselines3.readthedocs.io/)
- [Gymnasium Documentation](https://gymnasium.farama.org/)
