#!/bin/bash

# Variables
python_script="bert_inf.py"  # Replace with the name of your Python file
output_csv="timing_results.csv"
runs=50


for ((i=1; i<=runs; i++))
do
    result=$(python3 "$python_script" 1)
done

echo "Timing results collected in $output_csv"
