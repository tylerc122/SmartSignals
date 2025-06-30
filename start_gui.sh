echo "Starting SUMO GUI with Cross Intersection..."
echo "Make sure XQuartz is running!"

export DISPLAY=:0.0

# Start SUMO GUI with our cross intersection configuration
echo "🚀 Launching SUMO GUI..."
sumo-gui -c sumo_scenarios/cross_intersection.sumocfg

echo "✅ SUMO GUI closed" 