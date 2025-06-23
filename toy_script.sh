#!/bin/bash
# filepath: run_experiments.sh

# Create directories if they don't exist
mkdir -p figs

# Run the experiment 10 times
for i in {1..10}
do
    echo "Running experiment $i"
    python toy.py
    sleep 1  # Small delay to ensure unique timestamps
done