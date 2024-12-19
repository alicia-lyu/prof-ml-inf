import json
import pandas as pd
from sortedcontainers import SortedDict

def read_gpu_output(filename):
    with open(filename, 'r') as f:
        result = json.load(f)
    return result

gpu0_output = read_gpu_output("GPUtimeline_0.json")
gpu1_output = read_gpu_output("GPUtimeline_1.json")
gpu2_output = read_gpu_output("GPUtimeline_2.json")
gpu3_output = read_gpu_output("GPUtimeline_3.json")
pci_stats = pd.read_csv("./pci_stats/gpu_pcie_throughput.csv")

def sort_gpu_output(gpu_output):
    d = SortedDict()
    for layer_name, timings in gpu_output.items():
        for timing_pair in timings:
            d[(timing_pair[0], timing_pair[1])] = layer_name
    return d

q0 = sort_gpu_output(gpu0_output)    
q1 = sort_gpu_output(gpu1_output)    
q2 = sort_gpu_output(gpu2_output)    
q3 = sort_gpu_output(gpu3_output)

def traverse_gpu_output(d, pci_event_timing):
    relevant_layers = []
    for k, v in d.items():
        start_time, end_time = k
        if pci_event_timing >= start_time and pci_event_timing <= end_time:
            relevant_layers = [layer for layer in relevant_layers if not (layer in v)] # remove larger layer/module if a smaller layer can be found
            relevant_layers.append(v)
    return relevant_layers

# Identify interesting events in pci_stats and find relevent layers
data = {
    "time": [],
    "gpu_id": [],
    "to": [],
    "from": [],
    "layers": []
}
def process_gpu_output(i, gpu_id, pci_stats, gpu_dict):
    global data
    
    gpu_to = getattr(pci_stats, f"gpu{gpu_id}_to")
    gpu_from = getattr(pci_stats, f"gpu{gpu_id}_from")
    label = None
    if gpu_to > 100000 or gpu_from > 100000:
        label = 'major data transfer'
    elif gpu_to > 5000 or gpu_from > 20000:
        label = 'minor data transfer'
    else:
        return

    timing = pci_stats.time
    relevant_layers = traverse_gpu_output(gpu_dict, timing)
    print(f"{label}, line {i}, {pci_stats.time} | Relevant layers for GPU {gpu_id} (to: {gpu_to}, from: {gpu_from}): {relevant_layers}")
    
    data["time"].append(pci_stats.time)
    data["gpu_id"].append(gpu_id)
    data["from"].append(gpu_from)
    data["to"].append(gpu_to)
    data["layers"].append("|".join(relevant_layers))
    

gpu_dicts = [q0, q1, q2, q3]

for i, row in enumerate(pci_stats.itertuples()):
    for gpu_id, gpu_dict in enumerate(gpu_dicts):
        process_gpu_output(i, gpu_id, row, gpu_dict)
        
df = pd.DataFrame(data)
df.to_csv("pcie_analysis.csv", index=False)