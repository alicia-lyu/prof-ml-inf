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
    print("\\begin{tabular}{|l|c|c|c|c|c|c|c|}")
    print("\\hline")
    print("\\textbf{Kernel Name} & \\textbf{CPU-0 (ms)} & \\textbf{CPU-1 (ms)} & \\textbf{CPU\\% Var} & \\textbf{GPU-0 (ms)} & \\textbf{GPU-1 (ms)} & \\textbf{GPU\\% Var} & \\textbf{Calls} \\\\ \\hline")
    for i1, i2 in zip(l1,l2):
        # print(i1['Name'], i2["Name"])

        
        # print(i1)
        # print()
        # print(i2)
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
        if cuda_var != 0 or cpu_var!=0:
            var[i1['Name']] = {
                "CPU1 (us)": round(cpu1, 2),
                "CPU2 (us)": round(cpu2,2),
                "CPU_%Var": round(cpu_var,4),
                "GPU1 (us)": round(cuda1,2),
                "GPU2 (us)": round(cuda2,2),
                "GPU_%Var": round(cuda_var,4),
                "Calls": i1["Calls"]
                
            }
        name = i1['Name'].replace("_", "\\_")
        if cpu1 != 0 or cuda1 != 0:
            print( "\\textbf{"+  f"{name}" + "}" + f" & {round(cpu1,2)} & {round(cpu2,2)} & {round(cpu_var,2)} & {round(cuda1,2)} & {round(cuda2,2)} & {round(cuda_var,2)} & {i1['Calls']} \\\\ \hline")
    my_layer = layer.replace("_", "\\_")
    print("\\end{tabular}}")
    print("\\vspace{5pt}")
    print("\\centering")
    print("\\caption{" + f"Kernel breakdown for {my_layer}" + ".}")
    print("\\end{table*}")
    return var
            

# File paths for the two JSON files
file_path1 = 'data_0.json'
file_path2 = 'data_3.json'

# Load the JSON data into dictionaries
dict1 = load_json(file_path1)
dict2 = load_json(file_path2)

layers = ["encoder.block.0.layer.0.SelfAttention.q", "encoder.block.0.layer.0.SelfAttention.k", "encoder.block.0.layer.0.SelfAttention.v", "encoder.block.0.layer.0.SelfAttention.relative_attention_bias", "encoder.block.0.layer.0.SelfAttention.o"]
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
