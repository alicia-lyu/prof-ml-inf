import pynvml
import pandas as pd
import time

def get_pcie_throughput(handle):
    """Retrieve TX and RX throughput for a single GPU."""
    tx_throughput = pynvml.nvmlDeviceGetPcieThroughput(handle, pynvml.NVML_PCIE_UTIL_TX_BYTES)
    rx_throughput = pynvml.nvmlDeviceGetPcieThroughput(handle, pynvml.NVML_PCIE_UTIL_RX_BYTES)
    return tx_throughput, rx_throughput

def monitor_gpu_pcie_to_dataframe(interval=1.0, duration=10):
    """Monitor PCIe throughput for all GPUs and store the results in a DataFrame."""
    # Initialize NVML
    pynvml.nvmlInit()
    df = pd.DataFrame()  # Initialize an empty DataFrame
    data = {
        'gpu0-to': [],
        'gpu0-from': [],
        'gpu1-to': [],
        'gpu1-from': [],
        'gpu2-to': [],
        'gpu2-from': [],
        'gpu3-to': [],
        'gpu3-from': [],
    }
    
    try:
        # Get number of GPUs
        device_count = pynvml.nvmlDeviceGetCount()

        print(f"Monitoring PCIe Throughput for {device_count} GPUs...")
        
        # get a handle for each GPU
        handle0 = pynvml.nvmlDeviceGetHandleByIndex(0)
        handle1 = pynvml.nvmlDeviceGetHandleByIndex(1)
        handle2 = pynvml.nvmlDeviceGetHandleByIndex(2)
        handle3 = pynvml.nvmlDeviceGetHandleByIndex(3)
        start_time = time.time()
        
        while time.time() - start_time < duration:

            # get tx (data from gpu to host) and rx (data from host to gpu)
            # for each gpu
            tx, rx = get_pcie_throughput(handle0)
            data['gpu0-from'].append(tx)
            data['gpu0-to'].append(rx)
            tx, rx = get_pcie_throughput(handle1)
            data['gpu1-from'].append(tx)
            data['gpu1-to'].append(rx)
            tx, rx = get_pcie_throughput(handle2)
            data['gpu2-from'].append(tx)
            data['gpu2-to'].append(rx)
            tx, rx = get_pcie_throughput(handle3)
            data['gpu3-from'].append(tx)
            data['gpu3-to'].append(rx)
 
    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        # Shutdown NVML
        pynvml.nvmlShutdown()
       
        df = pd.DataFrame(data)

        print(df)
        print(df.median())
        print(df.mean().sort_values())
    return df

if __name__ == "__main__":
    # Monitor PCIe throughput and save to a DataFrame
    data = monitor_gpu_pcie_to_dataframe(interval=0.0, duration=10)

    # Optionally, save to a CSV file
    data.to_csv("gpu_pcie_throughput.csv", index=False)
