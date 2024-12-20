from transformers import LEDTokenizer, LEDForConditionalGeneration
import torch.nn as nn
import torch
import time
from functools import partial
import json
import fitz, sys
## Define hook functions

run_type = None
if len(sys.argv) > 1:
    run_type = int(sys.argv[1])
    print(f"Run type: {run_type}")
    # 1: 4 MPI, 20 runs, with hooks; 2: PCIe, with hooks 1 run; 3: single GPU, no hook, 4: 4 MPI, 20 runs, no hooks
    
start_events = {}
end_events = {}

take_time_dict = {}
count = 0
counters = {}
layer_timings = {}
from mpi4py import MPI

comm = MPI.COMM_WORLD
gpu_id = comm.Get_rank() % 4

torch.cuda.set_device(gpu_id)
device = torch.device(f"cuda:{gpu_id}")


io_t1 = time.time()

def extract_all_text(pdf_path):
    # Open the PDF document
    
    pdf_document = fitz.open(pdf_path)
    all_text = ""

    # Loop through all pages and concatenate text
    # print(range(len(pdf_document)))
    for page_number in range(len(pdf_document)):
        page = pdf_document[page_number]
        all_text += page.get_text() + "\n"  # Add a newline after each page
        #
    pdf_document.close()
    return all_text

# Usage
pdf_path = "pdfs/pipedream.pdf"
input_text = extract_all_text(pdf_path)

io_t2 = time.time()

setup_start = torch.cuda.Event(enable_timing=True, )
setup_end = torch.cuda.Event(enable_timing=True, )
setup_start.record()


model_name = "allenai/led-base-16384"
model = LEDForConditionalGeneration.from_pretrained(model_name).to(device)
tokenizer = LEDTokenizer.from_pretrained(model_name)


encode_start = torch.cuda.Event(enable_timing=True, )
encode_end = torch.cuda.Event(enable_timing=True, )
model_start = torch.cuda.Event(enable_timing=True, )
model_end = torch.cuda.Event(enable_timing=True, )
decode_start = torch.cuda.Event(enable_timing=True, )
decode_end = torch.cuda.Event(enable_timing=True, )

setup_end.record()

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



io_time = (io_t2 - io_t1) * 1000
setup_time = setup_start.elapsed_time(setup_end)
encode_time = encode_start.elapsed_time(encode_end)
model_time = model_start.elapsed_time(model_end)
decode_time = decode_start.elapsed_time(decode_end)

timing_data = {
    "IO": io_time,
    "ModelSetup": setup_time,
    "Tokenization": encode_time,
    "ModelInference": model_time,
    "De-Tokenization": decode_time
}
data_string = f"{io_time},{setup_time},{encode_time},{model_time},{decode_time}\n"

headers = "io,model_setup,tokenization,model,de-tokenization\n"
import os
# Filepath for the CSV
if run_type == 3:
    csv_filename = f"LEDtiming_single.csv"
else:
    csv_filename = f"LEDtiming_{gpu_id}.csv"

if not os.path.isfile(csv_filename):
    with open(csv_filename, mode="w", newline="") as csv_file:
        csv_file.write(headers)

# Write to the CSV file
with open(csv_filename, mode="a", newline="") as csv_file:
    print(f"Saving results to stage results to {csv_filename}")
    csv_file.write(data_string)

