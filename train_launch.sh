#!/bin/bash

# Define the name of the virtual environment directory
VENV_DIR="venv"

# Check if the virtual environment directory exists
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment in $VENV_DIR..."
    python3 -m venv $VENV_DIR
else
    echo "Virtual environment already exists in $VENV_DIR."
fi

# Activate the virtual environment
source $VENV_DIR/bin/activate

# Install dependencies from requirements.txt
echo "Installing dependencies from requirements.txt..."
pip install --upgrade pip
pip install -r requirements.txt

# Run the training script with all passed arguments
echo "Starting training..."
python main.py "$@"

# Deactivate the virtual environment after the script finishes
deactivate