from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch.nn as nn
import torch
import time
from functools import partial
import json
import fitz 
import pandas as pd
## Define hook functions

very_beginning = torch.cuda.Event(enable_timing=True, )
very_beginning.record()

counters = {}
layer_timings = {}
layer_starts_approx = {} # for absolute time recorded by time.time()
from mpi4py import MPI

comm = MPI.COMM_WORLD
gpu_id = comm.Get_rank() % 4

io_time = None
ocr_time = None

torch.cuda.set_device(gpu_id)
device = torch.device(f"cuda:{gpu_id}")
# print(f"Using device: {device}")

if gpu_id == 0:
    run_id = int(time.time() * 1000)
else:
    run_id = None
run_id = comm.bcast(run_id)

def take_time_pre(layer_name, module, input):
    global counters, layer_timings, gpu_id, layer_starts_approx

    # Initialize counters and timing storage for the layer
    if layer_name not in counters:
        counters[layer_name] = 0
        layer_timings[layer_name] = []
        layer_starts_approx[layer_name] = []

    # Increment the counter for this layer
    counters[layer_name] += 1

    # Record the start time
    start_event = torch.cuda.Event(enable_timing=True)
    start_event.record()

    # Store the start event for this invocation
    layer_timings[layer_name].append({"start_event": start_event, "end_event": None})
    layer_starts_approx[layer_name].append(time.time())

def take_time(layer_name, module, input, output):
    global layer_timings

    # Record the end time for the current invocation
    end_event = torch.cuda.Event(enable_timing=True)
    end_event.record()

    # Retrieve the most recent start event
    timing_entry = layer_timings[layer_name][-1]
    timing_entry["end_event"] = end_event


def extract_all_text(pdf_path):
    global io_time, ocr_time
    # Open the PDF document
    t1 = time.time()
    pdf_document = fitz.open(pdf_path)
    t2 = time.time()
    io_time = 1000 * (t2 - t1)
    all_text = ""

    # Loop through all pages and concatenate text
    t1 = time.time()
    # print(range(len(pdf_document)))
    for page_number in range(len(pdf_document)):
        page = pdf_document[page_number]
        all_text += page.get_text() + "\n"  # Add a newline after each page
        if page_number == 6:
            break
        

    t2 = time.time()
    ocr_time = 1000 * (t2 - t1)
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
for name, module in model.named_modules():
    if name == "":
       
        module.register_forward_pre_hook( partial(take_time_pre, "encoder-decoder-transition") )
        module.register_forward_hook( partial(take_time, "encoder-decoder-transition") )
    else:
        module.register_forward_pre_hook( partial(take_time_pre, name) )
        module.register_forward_hook( partial(take_time, name) )



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

summary_ids = model.generate(inputs)
model_end.record()

# Decode the model output
decode_start.record()
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
decode_end.record()

# Ensure all events are complete
torch.cuda.synchronize()

layer_invocation_times = {}
layer_timings_abs = {}
for layer_name, timings in layer_timings.items():
    layer_invocation_times[layer_name] = []
    layer_timings_abs[layer_name] = []
    for timing_entry, start_approx in zip(timings, layer_starts_approx[layer_name]):
        if timing_entry["start_event"] and timing_entry["end_event"]:
            elapsed_time = timing_entry["start_event"].elapsed_time(timing_entry["end_event"])  # In milliseconds
            layer_invocation_times[layer_name].append(elapsed_time)
            layer_timings_abs[layer_name].append((start_approx, start_approx + elapsed_time / 1000))

output_file = f"GPUoutput_{gpu_id}.json"

with open(output_file, "w") as file:
    json.dump(layer_timings_abs, file, indent=4)

output_table = f"t5base_output_allGPUs.csv"
custom_separator = "|"
# Convert layer_invocation_times to a pandas DataFrame
data = {"gpu_id": gpu_id, "run_id": run_id}  # Initialize with GPU ID as the first column
for layer_name, timings in layer_invocation_times.items():
    # Join all timings for this layer using the custom separator
    data[layer_name] = custom_separator.join(map(str, timings))
df = pd.DataFrame([data])
df.to_csv(output_table, mode='a', index=False, header=not pd.io.common.file_exists(output_table))

encode_time = encode_start.elapsed_time(encode_end)
model_time = model_start.elapsed_time(model_end)
decode_time = decode_start.elapsed_time(decode_end)

timing_data = {
    "IO": io_time,
    "OCR": ocr_time,
    "Encode": encode_time,
    "Model": model_time,
    "Decode": decode_time
}
data_string = f"{io_time},{ocr_time},{encode_time},{model_time},{decode_time}"

headers = data_string.strip("{}").split(",")
import csv
# Filepath for the CSV
csv_filename = f"timing_{gpu_id}.csv"

# Write to the CSV file
with open(csv_filename, mode="a", newline="") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(headers)
# print(timing_data)
# Write to a JSON file
# output_path = "timing_results.json"
# with open(output_path, "w", encoding="utf-8") as json_file:
#     json.dump(timing_data, json_file, indent=4)

# print("Summary:")
# print(summary)

