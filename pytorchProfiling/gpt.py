import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.profiler import profile, ProfilerActivity, record_function

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
                # torch.cuda.synchronize()  # Synchronize CUDA operations
                pass
            self.log_results(prof, layer_type, layer_name)
            self.is_profiling = False  # Reset profiling flag
        return hook

    def log_results(self, profiler_result, layer_type, layer_name):
        with open(self.log_file, "a") as f:
            f.write(f"Layer Type: {layer_type}\n")
            f.write(f"Layer Name: {layer_name}\n")
            f.write(profiler_result.key_averages().table(
                sort_by="cuda_time_total",
                row_limit=10
            ))
            f.write("\nGPU Kernels:\n")
            for event in profiler_result.events():
                if event.device_type == "cuda":
                    f.write(f"{event.name}: {event.cuda_time_total:.3f}us\n")
            f.write("\n" + "=" * 80 + "\n")

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

# Example usage
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load T5 model and tokenizer
    model_name = "t5-base"
    model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
    tokenizer = T5Tokenizer.from_pretrained(model_name)

    # Create the profiler
    profiler = LayerProfiler(model, log_file="layer_gpu_profile.log")

    # Input text
    input_text = "The quick brown fox jumps over the lazy dog. Summarize this sentence."
    inputs = tokenizer.encode(input_text, return_tensors="pt", padding=True).to(device)

    # Generate summary
    summary_ids = model.generate(inputs, max_length=20, min_length=5, num_beams=4)

    # Decode the model output
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    # Remove hooks after profiling
    profiler.remove_hooks()

    print("Summary:", summary)
    print("Profiling completed. Check the log file for details.")
