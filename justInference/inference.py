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

# from mpi4py import MPI

# comm = MPI.COMM_WORLD
# gpu_id = comm.Get_rank() % 4
gpu_id = 0
io_time = None
ocr_time = None

torch.cuda.set_device(gpu_id)
device = torch.device(f"cuda:{gpu_id}")


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
pdf_path = "../pdfs/pipedream.pdf"
input_text = "summarize: " + extract_all_text(pdf_path)


# Load T5-small model and tokenizer
model_name = "t5-base"
model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
tokenizer = T5Tokenizer.from_pretrained(model_name)


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

