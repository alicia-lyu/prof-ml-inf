import torch
import torch.nn as nn
import torch.optim as optim
from torch.profiler import profile, ProfilerActivity

# Define a simple PyTorch model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(100, 50)
        

    def forward(self, x):
        x = self.fc1(x)
        return x

# Initialize the model, move it to GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = SimpleModel().to(device)

# Create random input data
inputs = torch.randn(64, 100).to(device)  # Batch size 64, input size 100

# Define a loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Profiling the training loop
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    # Forward pass
    outputs = model(inputs)

for event in prof.events():
    if str(event.device_type) == "DeviceType.CUDA":
        # print()
        print("GPU: ", event.name, event.cuda_time)
        # print(event.cuda_time)
    else:
        # print("CPU")
        print("CPU: ", event.name, event.cpu_time)

    print()