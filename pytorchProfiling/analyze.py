import json

# Function to load JSON from a file into a dictionary
def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

# File paths for the two JSON files
file_path1 = 'data_0.json'
file_path2 = 'data_3.json'

# Load the JSON data into dictionaries
dict1 = load_json(file_path1)
dict2 = load_json(file_path2)

# Print the dictionaries
l1 = dict1['encoder.block.0.layer.0.SelfAttention.relative_attention_bias']
l2 = dict2['encoder.block.0.layer.0.SelfAttention.relative_attention_bias']
for i1, i2 in zip(l1, l2):
    
    cpu1 = i1['CPU Time Total (us)']
    cpu2 = i2['CPU Time Total (us)']
    cpu_var = 0
    if cpu1 != 0:
        cpu_var = 100 *(abs(cpu1 - cpu2) / ((cpu1 + cpu2) / 2))
    
    cuda1 = i1['CUDA Time Total (us)']
    cuda2 = i2['CUDA Time Total (us)']
    cuda_var = 0
    if cuda1 != 0:
        cuda_var = 100 *(abs(cuda1 - cuda2) / ((cuda1 + cuda2) / 2))
    if cuda_var != 0:
        print(i1['Name'])
        print(cpu1, cpu2, cpu_var)
        print(cuda1, cuda2, cuda_var)
        print()
# print("Dictionary 2:", dict2)
