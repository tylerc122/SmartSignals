#!/bin/bash

# Smart Traffic Signals - GUI Launcher
# This script reliably starts SUMO GUI with our cross intersection

echo "🚦 Starting SUMO GUI with Cross Intersection..."
echo "Make sure XQuartz is running!"

# Set display for X11 forwarding
export DISPLAY=:0.0

# Navigate to project directory (in case script is run from elsewhere)
cd "$(dirname "$0")"

# Start SUMO GUI with our cross intersection configuration
echo "🚀 Launching SUMO GUI..."
sumo-gui -c sumo_scenarios/cross_intersection.sumocfg

echo "✅ SUMO GUI closed" 