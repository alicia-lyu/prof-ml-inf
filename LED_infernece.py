from transformers import LEDTokenizer, LEDForConditionalGeneration
import torch.nn as nn
import torch
import time
from functools import partial
import json
import fitz 
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


# def take_time_pre(layer_name, module, input):
#     global counters, layer_timings, gpu_id

#     # Initialize counters and timing storage for the layer
#     if layer_name not in counters:
#         counters[layer_name] = 0
#         layer_timings[layer_name] = []

#     # Increment the counter for this layer
#     counters[layer_name] += 1

#     # Record the start time
#     start_event = torch.cuda.Event(enable_timing=True)
#     start_event.record()

#     # Store the start event for this invocation
#     layer_timings[layer_name].append({"start_event": start_event, "end_event": None})

# def take_time(layer_name, module, input, output):
#     global layer_timings

#     # Record the end time for the current invocation
#     end_event = torch.cuda.Event(enable_timing=True)
#     end_event.record()

#     # Retrieve the most recent start event
#     timing_entry = layer_timings[layer_name][-1]
#     timing_entry["end_event"] = end_event


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
        #
        

    t2 = time.time()
    ocr_time = 1000 * (t2 - t1)
    pdf_document.close()
    return all_text

# Usage
pdf_path = "pdfs/pipedream.pdf"
input_text = extract_all_text(pdf_path)


# Load T5-small model and tokenizer
model_name = "allenai/led-base-16384"
model = LEDForConditionalGeneration.from_pretrained(model_name).to(device)
tokenizer = LEDTokenizer.from_pretrained(model_name)

# x = False
# count = 0
# for name, module in model.named_modules():
#     if name == "":
       
#         module.register_forward_pre_hook( partial(take_time_pre, "encoder-decoder-transition") )
#         module.register_forward_hook( partial(take_time, "encoder-decoder-transition") )
#     else:
#         module.register_forward_pre_hook( partial(take_time_pre, name) )
#         module.register_forward_hook( partial(take_time, name) )



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
inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=16384, truncation=True).to(device) 
encode_end.record()

# Generate the summary
model_start.record()

summary_ids = model.generate(inputs, max_length=1024, min_length=1, num_beams=4)
model_end.record()

# Decode the model output
decode_start.record()
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
decode_end.record()

# Ensure all events are complete
torch.cuda.synchronize()
gpu_layer_times = {}

# for layer_name in start_events.keys():
#     elapsed_time = start_events[layer_name].elapsed_time(end_events[layer_name])  # Convert ms to seconds
#     gpu_layer_times[layer_name] = elapsed_time

# layer_invocation_times = {}
# for layer_name, timings in layer_timings.items():
#     layer_invocation_times[layer_name] = []
#     for timing_entry in timings:
#         if timing_entry["start_event"] and timing_entry["end_event"]:
#             elapsed_time = timing_entry["start_event"].elapsed_time(timing_entry["end_event"])  # In milliseconds
#             layer_invocation_times[layer_name].append(elapsed_time)
        

# output_file = "output.json"
# with open(output_file, "w") as file:
# #     json.dump(take_time_dict, file, indent=4)
# output_file = f"GPUoutput_{gpu_id}.json"


# with open(output_file, "w") as file:
#     json.dump(layer_invocation_times, file, indent=4)


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
# print(data_string)
print(f"Rank {gpu_id}: {io_time + ocr_time + encode_time+ model_time + decode_time}")
# headers = data_string.strip("{}").split(",")
# import csv
# # Filepath for the CSV
# csv_filename = f"timing_{gpu_id}.csv"

# # Write to the CSV file
# with open(csv_filename, mode="a", newline="") as csv_file:
#     writer = csv.writer(csv_file)
#     writer.writerow(headers)
# # print(timing_data)
# Write to a JSON file
# output_path = "timing_results.json"
# with open(output_path, "w", encoding="utf-8") as json_file:
#     json.dump(timing_data, json_file, indent=4)

# print("Summary:")
# print(summary)

