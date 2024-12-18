#!/bin/bash

# Variables
python_script="inference.py"  # Replace with the name of your Python file
output_csv="timing_results.csv"
runs=20

# Initialize CSV file with header
# echo "Run,IO,OCR,Encode,Model,Decode" > "$output_csv"

# Loop to execute the Python file multiple times
for ((i=1; i<=runs; i++))
do
    # Run the Python script and capture the output
    result=$(mpirun -np 4 python3 "$python_script" 1)
    # echo $result
    # exit
    # Extract the timing results (assuming they follow the format: "Timing: value1,value2,...")
    # if [[ $result == Timing:* ]]; then
    #     timings=${result#Timing: }  # Remove "Timing: " prefix
    #     echo "$i,$timings" >> "$output_csv"  # Write the run number and timings to the CSV file
    # else
    #     echo "Error: Unexpected output from Python script on run $i" >&2
    # fi
done

echo "Timing results collected in $output_csv"
