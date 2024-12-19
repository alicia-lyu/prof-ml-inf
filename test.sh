#!/bin/bash

# Variables
python_script="inference.py"  # Replace with the name of your Python file
output_csv="timing_results.csv"
runs=20

rm t5base_output_allGPUs.csv

for ((i=1; i<=runs; i++))
do
    result=$(mpirun -np 4 python3 "$python_script" 1)
done

echo "Timing results collected in $output_csv"
