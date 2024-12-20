import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.profiler import profile, ProfilerActivity, record_function
import time
import torch
from functools import partial
import json
import fitz 
io_time = None
ocr_time = None
problems = {}

device = "cuda:0"
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



pdf_path = "../pdfs/pipedream.pdf"
input_text = "summarize: " + extract_all_text(pdf_path)


# Load T5-small model and tokenizer
model_name = "t5-base"
model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
tokenizer = T5Tokenizer.from_pretrained(model_name)


# encode_start = torch.cuda.Event(enable_timing=True, )
# encode_end = torch.cuda.Event(enable_timing=True, )
# model_start = torch.cuda.Event(enable_timing=True, )
# model_end = torch.cuda.Event(enable_timing=True, )
# decode_start = torch.cuda.Event(enable_timing=True, )
# decode_end = torch.cuda.Event(enable_timing=True, )

inputs = tokenizer.encode(input_text, padding=True, return_tensors="pt").to(device) 
with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                with_stack=True,
                profile_memory=True
            ) as prof:
    # # Tokenize the input text
    # encode_start.record()
    # encode_end.record()

    # Generate the summary
    # model_start.record()

    summary_ids = model.generate(inputs, max_length=999999, min_length=1)
    # model_end.record()

    # Decode the model output
    # decode_start.record()
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    # decode_end.record()

    # Ensure all events are complete
    torch.cuda.synchronize()

cpu_total = 0
gpu_total =0 
for event in prof.events():
    if str(event.device_type) == "DeviceType.CPU":
        cpu_total += event.cpu_time
    else:
        gpu_total += event.cuda_time 
    # print(f"{event.device_type}: ", event.name, event.cuda_time, event.cpu_time)
    # # if str(event.device_type) == "DeviceType.CUDA":
    # #     # print()
    # #     print("GPU: ", event.name, event.cuda_time)
    # #     # print(event.cuda_time)
    # # else:
    # #     # print("CPU")
    # #     print("CPU: ", event.name, event.cpu_time)

    # print()
print(cpu_total, gpu_total)