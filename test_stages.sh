#!/bin/bash

# Variables
python_script="inference.py"  # Replace with the name of your Python file
runs=20

rm timing_*.csv

for ((i=1; i<=runs; i++))
do
    result=$(python3 "$python_script" 3)
done


for ((i=1; i<=runs; i++))
do
    result=$(mpirun -np 4 python3 "$python_script" 4)
done