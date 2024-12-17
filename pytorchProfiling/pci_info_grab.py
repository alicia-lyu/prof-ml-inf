import pynvml

# Initialize NVML
pynvml.nvmlInit()

# Get handle for GPU 0
gpu_index = 0
handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)

# Retrieve PCIe information
pci_info = pynvml.nvmlDeviceGetPciInfo(handle)
tx_throughput = pynvml.nvmlDeviceGetPcieThroughput(handle, pynvml.NVML_PCIE_UTIL_TX_BYTES)
rx_throughput = pynvml.nvmlDeviceGetPcieThroughput(handle, pynvml.NVML_PCIE_UTIL_RX_BYTES)
max_link_gen = pynvml.nvmlDeviceGetMaxPcieLinkGeneration(handle)
max_link_width = pynvml.nvmlDeviceGetMaxPcieLinkWidth(handle)
curr_link_gen = pynvml.nvmlDeviceGetCurrPcieLinkGeneration(handle)
curr_link_width = pynvml.nvmlDeviceGetCurrPcieLinkWidth(handle)

# Display PCIe metrics
print(f"GPU {gpu_index} PCI Info:")
print(f"  Bus ID: {pci_info.busId}")
print(f"  Device ID: {pci_info.device}")
print(f"  Domain ID: {pci_info.domain}")
print(f"  TX Throughput: {tx_throughput} KB/s")
print(f"  RX Throughput: {rx_throughput} KB/s")
print(f"  Max Link Generation: PCIe Gen {max_link_gen}")
print(f"  Max Link Width: {max_link_width} lanes")
print(f"  Current Link Generation: PCIe Gen {curr_link_gen}")
print(f"  Current Link Width: {curr_link_width} lanes")

# Shutdown NVML
pynvml.nvmlShutdown()
