import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.profiler import profile, ProfilerActivity, record_function
import time
import torch
from functools import partial
import json
import fitz 
gpu_id = 0
io_time = None
ocr_time = None
problems = {}


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

class LayerProfiler:
    def __init__(self, model, log_file="layer_profile.log", problem_layers=None):
        self.model = model
        self.log_file = log_file
        self.hooks = []
        self.is_profiling = False  # To prevent recursion
        self.problem_layers = problem_layers
        self.layer_kernel_dict = {}
        # Attach hooks to each layer
        for name, module in model.named_modules():
            if not isinstance(module, torch.nn.Sequential) and not isinstance(module, torch.nn.ModuleList):
                hook = module.register_forward_hook(self.forward_hook(name, module))
                self.hooks.append(hook)

    def forward_hook(self, layer_name, module):
        global problems
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
                
                """
                TODO: make more elegant - this either uses a list of problem layers to avoid 
                (layers without the forward method implemented b/c they are modules) or 
                catches if it is a high level module through a try-accept.  
                This is not ideal, however, this is the current model 
                agnostic solution.
                """
                if self.problem_layers:
                    if layer_name in self.problem_layers:
                        pass
                    else:
                        module(*input)
                else:
                    try:
                        module(*input)
                    except:
                       pass
          
           
            self.log_results(prof, layer_type, layer_name)
            self.is_profiling = False  # Reset profiling flag
        return hook

    def log_results(self, profiler_result, layer_type, layer_name):
        table = profiler_result.key_averages()
        results = []

        for event in table:
            entry = {
                "Name": event.key,  # Operation name
                "CPU Time Total (us)": event.cpu_time_total if hasattr(event, "cpu_time_total") else 0,
                "CUDA Time Total (us)": event.device_time_total if hasattr(event, "device_time_total") else 0,
                "Calls": event.count if hasattr(event, "count") else 0,
                "Self CPU Time Total (us)": event.self_cpu_time_total if hasattr(event, "self_cpu_time_total") else 0,
                "Self CUDA Time Total (us)": event.self_device_time_total if hasattr(event, "self_device_time_total") else 0,
                "CPU Memory Used (bytes)": event.cpu_memory_usage if hasattr(event, "cpu_memory_usage") else 0,
                "CUDA Memory Used (bytes)": event.self_device_memory_usage if hasattr(event, "self_device_memory_usage") else 0,
                "Input Shapes": event.input_shapes if hasattr(event, "input_shapes") else None,
            }

            if layer_name in self.layer_kernel_dict.keys():
                    self.layer_kernel_dict[layer_name].append(entry)
            else: 
                self.layer_kernel_dict[layer_name] = [entry]
            # break
        # with open(self.log_file, "a") as f:
        #     f.write(f"Layer Type: {layer_type}\n")
        #     f.write(f"Layer Name: {layer_name}\n")
        #     f.write(profiler_result.key_averages().table(
        #         sort_by="cuda_time_total",
        #         row_limit=None
        #     ))
            # print(results)
            # exit()
                # exit()
                # print(event.use_device)
                # if "cuda" in attr:
                #     print(attr)
                #     exit()
                # results.append({
            # "Name": event.key,
            # "CPU Time Total (us)": event.cpu_time_total if "cpu_time_total" in attr else 0,
            # "CUDA Time Total (us)": event.cuda_time_total if "cuda_time_total" in attr else 0,
            # "Calls": event.count if "count" in attr else 0,
            # "Self CPU Time Total (us)": event.self_cpu_time_total if "self_cpu_time_total" in attr else 0,
            # "Self CUDA Time Total (us)": event.self_cuda_time_total if "self_cuda_time_total" in attr else 0,
            # "CPU Memory Used (bytes)": event.cpu_memory_usage if hasattr(event, "cpu_memory_usage") else 0,
            # "CUDA Memory Used (bytes)": event.cuda_memory_usage if hasattr(event, "cuda_memory_usage") else 0,
            # })
            # print(results)
            # exit()
            # f.write("\nGPU Kernels:\n")
            # for event in profiler_result.events():
            #     if event.device_type == "cuda":
            #         f.write(f"{event.name}: {event.cuda_time_total:.3f}us\n")
            # f.write("\n" + "=" * 80 + "\n")

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

# Example usage
if __name__ == "__main__":
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    gpu_id = comm.Get_rank() % 4
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(gpu_id)
    # print(device)
    # exit()
    # Load T5 model and tokenizer
    model_name = "t5-base"
    model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
    tokenizer = T5Tokenizer.from_pretrained(model_name)


    annoying = ['encoder.block.0.layer.0.SelfAttention', 'encoder.block.0.layer.0', 'encoder.block.0', 'encoder.block.1.layer.0.SelfAttention', 'encoder.block.1.layer.0', 'encoder.block.1', 'encoder.block.2.layer.0.SelfAttention', 'encoder.block.2.layer.0', 'encoder.block.2', 'encoder.block.3.layer.0.SelfAttention', 'encoder.block.3.layer.0', 'encoder.block.3', 'encoder.block.4.layer.0.SelfAttention', 'encoder.block.4.layer.0', 'encoder.block.4', 'encoder.block.5.layer.0.SelfAttention', 'encoder.block.5.layer.0', 'encoder.block.5', 'encoder.block.6.layer.0.SelfAttention', 'encoder.block.6.layer.0', 'encoder.block.6', 'encoder.block.7.layer.0.SelfAttention', 'encoder.block.7.layer.0', 'encoder.block.7', 'encoder.block.8.layer.0.SelfAttention', 'encoder.block.8.layer.0', 'encoder.block.8', 'encoder.block.9.layer.0.SelfAttention', 'encoder.block.9.layer.0', 'encoder.block.9', 'encoder.block.10.layer.0.SelfAttention', 'encoder.block.10.layer.0', 'encoder.block.10', 'encoder.block.11.layer.0.SelfAttention', 'encoder.block.11.layer.0', 'encoder.block.11', 'encoder', 'decoder.block.0.layer.0.SelfAttention', 'decoder.block.0.layer.0', 'decoder.block.0.layer.1.EncDecAttention', 'decoder.block.0.layer.1', 'decoder.block.0', 'decoder.block.1.layer.0.SelfAttention', 'decoder.block.1.layer.0', 'decoder.block.1.layer.1.EncDecAttention', 'decoder.block.1.layer.1', 'decoder.block.1', 'decoder.block.2.layer.0.SelfAttention', 'decoder.block.2.layer.0', 'decoder.block.2.layer.1.EncDecAttention', 'decoder.block.2.layer.1', 'decoder.block.2', 'decoder.block.3.layer.0.SelfAttention', 'decoder.block.3.layer.0', 'decoder.block.3.layer.1.EncDecAttention', 'decoder.block.3.layer.1', 'decoder.block.3', 'decoder.block.4.layer.0.SelfAttention', 'decoder.block.4.layer.0', 'decoder.block.4.layer.1.EncDecAttention', 'decoder.block.4.layer.1', 'decoder.block.4', 'decoder.block.5.layer.0.SelfAttention', 'decoder.block.5.layer.0', 'decoder.block.5.layer.1.EncDecAttention', 'decoder.block.5.layer.1', 'decoder.block.5', 'decoder.block.6.layer.0.SelfAttention', 'decoder.block.6.layer.0', 'decoder.block.6.layer.1.EncDecAttention', 'decoder.block.6.layer.1', 'decoder.block.6', 'decoder.block.7.layer.0.SelfAttention', 'decoder.block.7.layer.0', 'decoder.block.7.layer.1.EncDecAttention', 'decoder.block.7.layer.1', 'decoder.block.7', 'decoder.block.8.layer.0.SelfAttention', 'decoder.block.8.layer.0', 'decoder.block.8.layer.1.EncDecAttention', 'decoder.block.8.layer.1', 'decoder.block.8', 'decoder.block.9.layer.0.SelfAttention', 'decoder.block.9.layer.0', 'decoder.block.9.layer.1.EncDecAttention', 'decoder.block.9.layer.1', 'decoder.block.9', 'decoder.block.10.layer.0.SelfAttention', 'decoder.block.10.layer.0', 'decoder.block.10.layer.1.EncDecAttention', 'decoder.block.10.layer.1', 'decoder.block.10', 'decoder.block.11.layer.0.SelfAttention', 'decoder.block.11.layer.0', 'decoder.block.11.layer.1.EncDecAttention', 'decoder.block.11.layer.1', 'decoder.block.11', 'decoder', '']
    # Create the profiler
    profiler = LayerProfiler(model, log_file="layer_gpu_profile.log", problem_layers = annoying)
    pdf_path = "../pdfs/pipedream.pdf"
    input_text = "summarize: " + extract_all_text(pdf_path)

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

    # Remove hooks after profiling
    profiler.remove_hooks()
    print("Runtime: ", io_time + ocr_time + encode_time + model_time + decode_time)
    # print("Summary:", summary)
    print("Profiling completed. Check the log file for details.")
    # print(data_string)
    # print()
    with open(f"data_{gpu_id}.json", "w") as json_file:
        json.dump(profiler.layer_kernel_dict, json_file, indent=4)
    
