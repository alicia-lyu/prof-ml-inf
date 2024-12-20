import json
import pandas as pd
# Function to load JSON from a file into a dictionary
def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def extractData(dict1, dict2, layer):
    cpu_total = 0
    cpu1_total = 0
    gpu_total = 0
    gpu1_total = 0
    l1 = sorted(dict1[layer], key=lambda x: x['Name'])
    l2 = sorted(dict2[layer], key=lambda x: x['Name'])
    var = {}
    for i1, i2 in zip(l1,l2):


        cpu1 = i1['CPU Time (us)']
        cpu2 = i2['CPU Time (us)']
        cpu_total += i1['CPU Time (us)']
        cpu1_total += i2['CPU Time (us)']
        cpu_var = 0
        if cpu1 != 0:
            cpu_var = 100 *(abs(cpu1 - cpu2) / ((cpu1 + cpu2) / 2))
        
        cuda1 = i1['CUDA Time (us)']
        gpu_total += i1['CUDA Time (us)']
        gpu1_total += i2['CUDA Time (us)']
        cuda2 = i2['CUDA Time (us)']
        cuda_var = 0
        if cuda1 != 0:
            cuda_var = 100 *(abs(cuda1 - cuda2) / ((cuda1 + cuda2) / 2))
        if cuda_var != 0 or cpu_var!=0:
            var[i1['Name']] = {
                "CPU1 (us)": round(cpu1, 2),
                "CPU2 (us)": round(cpu2,2),
                "CPU_%Var": round(cpu_var,4),
                "GPU1 (us)": round(cuda1,2),
                "GPU2 (us)": round(cuda2,2),
                "GPU_%Var": round(cuda_var,4),                
            }
    return var, cpu_total, gpu_total, cpu1_total, gpu1_total
            

# File paths for the two JSON files
file_path1 = 'data_0.json'
file_path2 = 'data_3.json'

# Load the JSON data into dictionaries
dict1 = load_json(file_path1)
dict2 = load_json(file_path2)

layers = ["encoder.block.0.layer.0.layer_norm", "encoder.block.0.layer.0.SelfAttention.q", "encoder.block.0.layer.0.SelfAttention.k", "encoder.block.0.layer.0.SelfAttention.v", "encoder.block.0.layer.0.SelfAttention.relative_attention_bias", "encoder.block.0.layer.0.SelfAttention.o", "encoder.block.0.layer.0.dropout"]
layers = dict1.keys()
# layers = ["encoder.block.3.layer.0.SelfAttention.k", "encoder.block.3.layer.0.SelfAttention.o", "encoder.block.3.layer.0.SelfAttention.v","encoder.block.3.layer.0.SelfAttention.q"]
# layers = ["encoder.block.3.layer.0.SelfAttention.q"]

gpu_total = 0
gpu1_total = 0
cpu1_total = 0
cpu_total = 0
for layer in layers:
    my_var, cpu, gpu, cpu1, gpu1 = extractData(dict1, dict2,layer)
    df = pd.DataFrame.from_dict(my_var, orient='index')
    print(df)
    print()
    cpu_total += cpu
    gpu_total += gpu
    cpu1_total += cpu1
    gpu1_total += gpu1
    
# Print the dictionaries
print(gpu_total / 1000, cpu_total / 1000)
print(gpu1_total / 1000, cpu1_total / 1000)
# print("Dictionary 2:", dict2)
