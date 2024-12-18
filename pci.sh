#!/bin/bash

python_script="inference.py"  # Replace with the name of your Python file
echo "t0: $(date '+%Y-%m-%d %H:%M:%S')"
(mpirun -np 4 python3 "$python_script" 2; echo "mpirun complete: $(date '+%Y-%m-%d %H:%M:%S')")&
cd "./pci_stats"
echo "t1: $(date '+%Y-%m-%d %H:%M:%S')"
python3 "pci_monitor.py"