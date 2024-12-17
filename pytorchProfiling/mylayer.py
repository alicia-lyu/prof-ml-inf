from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch.nn as nn
import torch
import time
from functools import partial
import json
import fitz 
import pandas as pd

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

import subprocess

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

layer = model.encoder.block[0].layer[0]

# Dummy input for profiling
batch_size = 1
seq_length = 8
hidden_dim = model.config.d_model  # T5 hidden size
dummy_input = torch.randn(batch_size, seq_length, hidden_dim).to("cuda")

# Enable model to use CUDA
model.to("cuda")
layer.to("cuda")

# Profile the kernel launches
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    on_trace_ready=torch.profiler.tensorboard_trace_handler("./profiler_output"),
    with_stack=True,
    record_shapes=True
) as prof:
    # Forward pass through the layer
    output = layer(dummy_input)

# Print the profiler results
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))