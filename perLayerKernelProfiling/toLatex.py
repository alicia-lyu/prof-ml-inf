import json
import pandas as pd
# Function to load JSON from a file into a dictionary
def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def extractData(dict1, dict2, layer):
    l1 = sorted(dict1[layer], key=lambda x: x['Name'])
    l2 = sorted(dict2[layer], key=lambda x: x['Name'])
    var = {}
    print("\\begin{table*}[h!]")
    print("\\centering")
    print("\\scalebox{0.9}{")
    print("\\begin{tabular}{|l|c|c|c|c|c|c|}")
    print("\\hline")
    print("\\textbf{Kernel Name} & \\textbf{CPU-0 (ms)} & \\textbf{CPU-1 (ms)} & \\textbf{CPU\\% Var} & \\textbf{GPU-0 (ms)} & \\textbf{GPU-1 (ms)} & \\textbf{GPU\\% Var} \\\\ \\hline")
    for i1, i2 in zip(l1,l2):
        # print(i1['Name'], i2["Name"])

        
        # print(i1)
        # print()
        # print(i2)
        
        cpu1 = round(i1['CPU Time (us)'],2)
        cpu2 = round(i2['CPU Time (us)'],2)
        cpu_var = 0
        if cpu1 != 0:
            cpu_var = round(100 *(abs(cpu1 - cpu2) / ((cpu1 + cpu2) / 2)),2)
        else:
            cpu1 = "N/A"
            cpu2 = "N/A"
            cpu_var = "N/A"
        cuda1 = round(i1['CUDA Time (us)'],2)
        cuda2 = round(i2['CUDA Time (us)'],2)
        cuda_var = 0
        if cuda1 != 0:
            cuda_var = round(100 *(abs(cuda1 - cuda2) / ((cuda1 + cuda2) / 2)),2)
        else:
            cuda1 = "N/A"
            cuda2 = "N/A"
            cuda_var = "N/A"
        name = i1['Name'].replace("_", "\\_")
        if cpu1 != 0 or cuda1 != 0:
            print( "\\textbf{"+  f"{name}" + "}" + f" & {cpu1} & {cpu2} & {cpu_var} & {cuda1} & {cuda2} & {cuda_var} \\\\ \hline")
    my_layer = layer.replace("_", "\\_")
    print("\\end{tabular}}")
    print("\\vspace{5pt}")
    print("\\centering")
    print("\\caption{" + f"Kernel breakdown for {my_layer}" + ".}")
    print("\\end{table*}")
    return var
            

# File paths for the two JSON files
file_path1 = 'blocking_data_0.json'
file_path2 = 'blocking_data_3.json'

# Load the JSON data into dictionaries
dict1 = load_json(file_path1)
dict2 = load_json(file_path2)

layers = ["encoder.block.0.layer.0.layer_norm", "encoder.block.0.layer.0.SelfAttention.q", "encoder.block.0.layer.0.SelfAttention.k", "encoder.block.0.layer.0.SelfAttention.v", "encoder.block.0.layer.0.SelfAttention.relative_attention_bias", "encoder.block.0.layer.0.SelfAttention.o", "encoder.block.0.layer.0.dropout"]
# layers = ["encoder.block.3.layer.0.SelfAttention.k", "encoder.block.3.layer.0.SelfAttention.o", "encoder.block.3.layer.0.SelfAttention.v","encoder.block.3.layer.0.SelfAttention.q"]
# layers = ["encoder.block.3.layer.0.SelfAttention.q"]

for layer in layers:
    my_var = extractData(dict1, dict2,layer)
    df = pd.DataFrame.from_dict(my_var, orient='index')
    print()
    # print(layer)
    # print("___________________________")
    # for _, row in df.iterrows():
    #     print()
        # print(row['CPU1 (us)'])
        # exit()
    # print(df)
    # print()
# Print the dictionaries

# print("Dictionary 2:", dict2)
