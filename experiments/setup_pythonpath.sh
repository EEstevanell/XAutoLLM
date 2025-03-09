#!/bin/bash

# Add the experiments directory to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:/home/coder/autogoal/experiments

echo "PYTHONPATH has been updated to include /home/coder/autogoal/experiments"
echo "You can now import modules using 'from src.data_loading.data_loader' etc."
echo "This setting is only active for the current terminal session."