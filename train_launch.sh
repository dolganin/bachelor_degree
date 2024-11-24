#!/bin/bash

# Get the script's directory
SCRIPT_DIR="$(dirname "$(realpath "$0")")"

# Define the name of the virtual environment directory
VENV_DIR="$SCRIPT_DIR/bd_env"

# Check if the virtual environment directory exists
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment in $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
fi

# Activate the virtual environment
source "$VENV_DIR/bin/activate"

# Check if reqs/nn_requirements.txt exists
REQ_FILE="$SCRIPT_DIR/reqs/nn_requirements.txt"
if [ ! -f "$REQ_FILE" ]; then
    echo "Error: $REQ_FILE not found."
    deactivate
    exit 1
fi

# Calculate the current hash of nn_requirements.txt
CURRENT_HASH=$(md5sum "$REQ_FILE" | awk '{print $1}')

# Check the stored hash
STORED_HASH_FILE="$VENV_DIR/.requirements_hash"
if [ -f "$STORED_HASH_FILE" ]; then
    STORED_HASH=$(cat "$STORED_HASH_FILE")
else
    STORED_HASH=""
fi

# Compare hashes
if [ "$CURRENT_HASH" != "$STORED_HASH" ]; then
    echo "Installing dependencies from $REQ_FILE..."
    pip install --upgrade pip
    pip install -r "$REQ_FILE" || { echo "Error installing dependencies."; deactivate; exit 1; }
    echo "$CURRENT_HASH" > "$STORED_HASH_FILE"
else
    echo "Packages already installed and requirements have not changed."
fi

# Check if DoomITH repository is already cloned
DOOMITH_DIR="$SCRIPT_DIR/DoomITH"
if [ ! -d "$DOOMITH_DIR/.git" ]; then
    echo "Cloning DoomITH repository..."
    git clone https://github.com/dolganin/DoomITH.git "$DOOMITH_DIR"
    echo "Building and installing DoomITH..."
    cd "$DOOMITH_DIR" || exit
    mkdir -p build
    cd build || exit
    cmake ..
    make -j$(nproc)
    cd ..
    pip install .
fi
cd ../bachelor_degree

# Run the training script with all passed arguments
echo "Starting training..."
python main.py "$@"

# Deactivate the virtual environment after the script finishes
deactivate