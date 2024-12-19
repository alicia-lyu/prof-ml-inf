from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch.nn as nn
import torch
import time, sys
from functools import partial
import json
import fitz 
import pandas as pd
## Define hook functions


run_type = None
if len(sys.argv) > 1:
    run_type = int(sys.argv[1])
    print(f"Run type: {run_type}")
    # 1: 4 MPI, 20 runs, with hooks; 2: PCIe, with hooks 1 run; 3: single GPU, no hook, 4: 4 MPI, 20 runs, no hooks

counters = {}
layer_timings = {}
from mpi4py import MPI

num_gpus = torch.cuda.device_count()
if num_gpus == 0:
    raise RuntimeError("No GPUs detected. Please ensure a GPU is available.")
elif num_gpus == 1 or run_type == 3:
    print("Running on a single GPU.")
    gpu_id = 0 
else:
    print(f"Running on multiple GPUs ({num_gpus} detected).")
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    gpu_id = comm.Get_rank() % num_gpus

# Set GPU device
torch.cuda.set_device(gpu_id)
device = torch.device(f"cuda:{gpu_id}")
print(f"Using GPU {gpu_id} of {num_gpus} available GPUs.")

# Generate a unique run ID (shared across processes if using MPI)
if num_gpus > 1:
    run_id = None
    if gpu_id == 0:
        run_id = int(time.time() * 1000)
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    run_id = comm.bcast(run_id)
else:
    run_id = int(time.time() * 1000)

io_t1 = time.time()
print("io_t1: ", io_t1)
very_beginning = torch.cuda.Event(enable_timing=True, )
very_beginning_cpu_time = None

def take_time_pre(layer_name, module, input):
    global counters, layer_timings, gpu_id, very_beginning_cpu_time, very_beginning
    
    if len(counters.keys()) == 0:
        very_beginning.record()
        very_beginning_cpu_time = time.time()
        print(f'Very beginning: {very_beginning_cpu_time}')

    # Initialize counters and timing storage for the layer
    if layer_name not in counters:
        counters[layer_name] = 0
        layer_timings[layer_name] = []

    # Increment the counter for this layer
    counters[layer_name] += 1

    # Record the start time
    start_event = torch.cuda.Event(enable_timing=True)
    start_event.record()

    # Store the start event for this invocation
    layer_timings[layer_name].append({"start_event": start_event, "end_event": None})

def take_time(layer_name, module, input, output):
    global layer_timings

    # Record the end time for the current invocation
    end_event = torch.cuda.Event(enable_timing=True)
    end_event.record()

    # Retrieve the most recent start event
    timing_entry = layer_timings[layer_name][-1]
    timing_entry["end_event"] = end_event


def extract_all_text(pdf_path):
    # Open the PDF document
    
    pdf_document = fitz.open(pdf_path)
    all_text = ""

    # Loop through all pages and concatenate text
    # print(range(len(pdf_document)))
    for page_number in range(len(pdf_document)):
        page = pdf_document[page_number]
        all_text += page.get_text() + "\n"  # Add a newline after each page
        if page_number == 6:
            break
        
    pdf_document.close()
    return all_text

# Usage
pdf_path = "pdfs/pipedream.pdf"
input_text = "summarize: " + extract_all_text(pdf_path)


# Load T5-small model and tokenizer
model_name = "t5-base"
model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
tokenizer = T5Tokenizer.from_pretrained(model_name)

print("model.named_modules:", model.named_modules())
if run_type == 1 or run_type == 2:
    for name, module in model.named_modules():
        if name == "":  
            # module.register_forward_pre_hook( partial(take_time_pre, "encoder-decoder-transition") )
            # module.register_forward_hook( partial(take_time, "encoder-decoder-transition") )
            pass
        else:
            module.register_forward_pre_hook( partial(take_time_pre, name) )
            module.register_forward_hook( partial(take_time, name) )

io_t2 = time.time()
io_time = (io_t2 - io_t1) * 1000
print("io_t2:", io_t2, "io_time:", io_time)
# Prepend "summarize:" to the input text as required by T5
# input_text = "summarize: " + text_to_summarize

encode_start = torch.cuda.Event(enable_timing=True, )
encode_end = torch.cuda.Event(enable_timing=True, )
model_start = torch.cuda.Event(enable_timing=True, )
model_end = torch.cuda.Event(enable_timing=True, )
decode_start = torch.cuda.Event(enable_timing=True, )
decode_end = torch.cuda.Event(enable_timing=True, )


# Tokenize the input text
encode_start.record()
inputs = tokenizer.encode(input_text, padding=True, return_tensors="pt").to(device) 
encode_end.record()

# Generate the summary
model_start.record()

summary_ids = model.generate(inputs, max_length=999999, min_length=1)
model_end.record()

# Decode the model output
decode_start.record()
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
decode_end.record()

# Ensure all events are complete
torch.cuda.synchronize()

if run_type == 1 or run_type == 2:
    layer_invocation_times = {}
    layer_timings_abs = {}
    for layer_name, timings in layer_timings.items():
        layer_invocation_times[layer_name] = []
        layer_timings_abs[layer_name] = []
        for timing_entry in timings:
            if timing_entry["start_event"] and timing_entry["end_event"]:
                elapsed_time = timing_entry["start_event"].elapsed_time(timing_entry["end_event"])  # In milliseconds
                layer_invocation_times[layer_name].append(elapsed_time)
                layer_timings_abs[layer_name].append((
                    very_beginning.elapsed_time(timing_entry["start_event"]) / 1000 + very_beginning_cpu_time,
                    very_beginning.elapsed_time(timing_entry["end_event"])/ 1000 + very_beginning_cpu_time
                ))

if run_type == 2: # PCIe
    timeline_file = f"GPUtimeline_{gpu_id}.json"
    with open(timeline_file, "w") as file:
        json.dump(layer_timings_abs, file, indent=4)
else:
    print("Not updating timelines.")

if run_type == 1:
    output_table = f"t5base_output_allGPUs.csv"
    custom_separator = "|"
    # Convert layer_invocation_times to a pandas DataFrame
    data = {"gpu_id": gpu_id, "run_id": run_id}  # Initialize with GPU ID as the first column
    for layer_name, timings in layer_invocation_times.items():
        # Join all timings for this layer using the custom separator
        data[layer_name] = custom_separator.join(map(str, timings))
    df = pd.DataFrame([data])
    df.to_csv(output_table, mode='a', index=False, header=not pd.io.common.file_exists(output_table))
else:
    print("Not updating t5base_output_allGPUs.csv")

encode_time = encode_start.elapsed_time(encode_end)
model_time = model_start.elapsed_time(model_end)
decode_time = decode_start.elapsed_time(decode_end)

if run_type == 3 or run_type == 4:
    timing_data = {
        "IO": io_time,
        "Tokenization": encode_time,
        "Model": model_time,
        "De-Tokenization": decode_time
    }
    data_string = f"{io_time},{encode_time},{model_time},{decode_time}\n"

    headers = "io,tokenization,model,de-tokenization\n"
    import os
    # Filepath for the CSV
    if num_gpus == 1 or run_type == 3:
        csv_filename = f"timing_single.csv"
    else:
        csv_filename = f"timing_{gpu_id}.csv"

    if not os.path.isfile(csv_filename):
        with open(csv_filename, mode="w", newline="") as csv_file:
            csv_file.write(headers)

    # Write to the CSV file
    with open(csv_filename, mode="a", newline="") as csv_file:
        print(f"Saving results to stage results to {csv_filename}")
        csv_file.write(data_string)
else:
    print("Not updating stages timing.")