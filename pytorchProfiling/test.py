import torch
import torch.nn as nn
from torch.profiler import profile, ProfilerActivity, record_function
from transformers import T5ForConditionalGeneration, T5Tokenizer
import time
from functools import partial
import json
import fitz 
import pandas as pd
class LayerProfiler:
    def __init__(self, model, log_file="layer_profile.log"):
        self.model = model
        self.log_file = log_file
        self.hooks = []
        self.is_profiling = False  # To prevent recursion

        # Attach hooks to each layer
        for name, module in model.named_modules():
            if not isinstance(module, torch.nn.Sequential) and not isinstance(module, torch.nn.ModuleList):
                hook = module.register_forward_hook(self.forward_hook(name, module))
                self.hooks.append(hook)

    def forward_hook(self, layer_name, module):
        def hook(module, input, output):
            if self.is_profiling:
                return  # Prevent recursion
            self.is_profiling = True  # Set profiling flag
            layer_type = module.__class__.__name__
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                with_stack=True,
                profile_memory=True
            ) as prof:
                # Forward pass happens within the `generate` call
                torch.cuda.synchronize()  # Synchronize CUDA operations
            self.log_results(prof, layer_type, layer_name)
            self.is_profiling = False  # Reset profiling flag
        return hook

    def log_results(self, profiler_result, layer_type, layer_name):
        with open(self.log_file, "a") as f:
            f.write(f"Layer Type: {layer_type}\n")
            f.write(f"Layer Name: {layer_name}\n")
            f.write(profiler_result.key_averages().table(
                sort_by="cuda_time_total",
                row_limit=999999
            ))
            # f.write("\nGPU Kernels:\n")
            # for event in profiler_result.events():
            #     if event.device_type == "cuda":
            #         f.write(f"{event.name}: {event.cuda_time_total:.3f}us\n")
            f.write("\n" + "=" * 80 + "\n")

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


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

# Example usage
if __name__ == "__main__":
    # Define a sample model
   
    pdf_path = "pdfs/pipedream.pdf"
    input_text = "summarize: " + extract_all_text(pdf_path)

    device = "cuda:0"
    # Load T5-small model and tokenizer
    model_name = "t5-base"
    model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    # Create the profiler
    profiler = LayerProfiler(model, log_file="layer_gpu_profile.log")

    inputs = tokenizer.encode(input_text, padding=True, return_tensors="pt").to(device) 

    summary_ids = model.generate(inputs, max_length=2, min_length=1, num_beams=4)

    # Decode the model output
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    # # Run a dummy forward pass
    # dummy_input = torch.randn(10,16).to("cuda")
    # model(dummy_input)

    # Remove hooks after profiling
    # profiler.remove_hooks()

    print("Profiling completed. Check the log file for details.")
