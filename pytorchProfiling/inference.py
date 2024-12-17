from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch.nn as nn
import torch
import time
from functools import partial
import json
import fitz 
import pandas as pd
## Define hook functions
start_events = {}
end_events = {}

take_time_dict = {}
count = 0
counters = {}
layer_timings = {}
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

import pynvml

pynvml.nvmlInit()
  # Specify the GPU ID to monitor
handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)

def get_gpu_stats(handle):
    """Utility function to get GPU stats using PyNVML."""
    memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    util_rates = pynvml.nvmlDeviceGetUtilizationRates(handle)
    temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
    power_usage = pynvml.nvmlDeviceGetPowerUsage(handle)

    return {
        "memory_used_MB": memory_info.used / 1024 ** 2,
        "gpu_utilization_percent": util_rates.gpu,
        "temperature_C": temperature,
        "power_usage_W": power_usage / 1000.0,
    }

def take_time_pre(layer_name, module, input):
    global counters, layer_timings, handle

    if layer_name not in counters:
        counters[layer_name] = 0
        layer_timings[layer_name] = []

    counters[layer_name] += 1

    # Record the start time and GPU stats
    start_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    start_gpu_stats = get_gpu_stats(handle)

    layer_timings[layer_name].append({
        "start_event": start_event,
        "end_event": None,
        "start_gpu_stats": start_gpu_stats,
        "end_gpu_stats": None,
    })

def take_time(layer_name, module, input, output):
    global layer_timings, handle

    # Record the end time and GPU stats
    end_event = torch.cuda.Event(enable_timing=True)
    end_event.record()
    end_gpu_stats = get_gpu_stats(handle)

    # Retrieve current layer's timing entry
    timing_entry = layer_timings[layer_name][-1]
    timing_entry.update({
        "end_event": end_event,
        "end_gpu_stats": end_gpu_stats,
    })

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

x = False
count = 0
# print("model.named_modules:", model.named_modules())
for name, module in model.named_modules():
    if name == "":
       
        module.register_forward_pre_hook( partial(take_time_pre, "encoder-decoder-transition") )
        module.register_forward_hook( partial(take_time, "encoder-decoder-transition") )
    else:
        module.register_forward_pre_hook( partial(take_time_pre, name) )
        module.register_forward_hook( partial(take_time, name) )




start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)
start_event.record()
inputs = tokenizer.encode(input_text, padding=True, return_tensors="pt").to(device) 

summary_ids = model.generate(inputs, max_length=2, min_length=1, num_beams=4)

# Decode the model output
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

end_event.record()
# Ensure all events are complete
torch.cuda.synchronize()
layer_invocation_times = {}
for layer_name, timings in layer_timings.items():
    layer_invocation_times[layer_name] = []
    for timing_entry in timings:
        if timing_entry["start_event"] and timing_entry["end_event"]:
            elapsed_time = timing_entry["start_event"].elapsed_time(timing_entry["end_event"])  # In milliseconds
            layer_invocation_times[layer_name].append((elapsed_time, timing_entry['start_gpu_stats'], timing_entry['end_gpu_stats']))

# Filter for GPU (CUDA) kernel events

with open(f"output_{gpu_id}.json", "w") as json_file:
    json.dump(layer_invocation_times, json_file, indent=4)  # `indent=4` makes the file pretty-printed

print(f"GPU {gpu_id}: ", start_event.elapsed_time(end_event))
# Print details of each GPU kernel
# for kernel in all_events:
#     if kernel.use_device == 'cuda':

        # print(f"Kernel Name: {kernel.name}- time: {kernel.device_time_total}")
        # print(f"CUDA Time: {kernel.device_time_total:.2f} us")
        # print(f"Input Shapes: {kernel.input_shapes}")
        # print(type(kernel.use_device))
        # print(dir(kernel))
        # if kernel.stack:
        #     print("Stack Trace:")
        #     print("\n".join(kernel.stack[:5]))  # Print the first 5 lines of the stack trace
        # print("=" * 50)
        # exit()

    
    #     # Extract the layer name

# events = prof.events()

# for event in events:
#     if event.name == "encoder.block.0.layer.0.layer_norm":
#         print(f"Name: {event.name}")
#         print(f"Device: {event.device_type}")  # CPU or CUDA
#         print(f"CPU Time: {event.cpu_time_total:.2f} us")
#         print(f"CUDA Time: {event.device_time_total:.2f} us")
#         print(f"Input Shapes: {event.input_shapes}")
#         print(event.kernels)
#         print(dir(event))
#         exit()
