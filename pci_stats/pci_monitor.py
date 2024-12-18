import pynvml
import pandas as pd
import time

def get_pcie_throughput(handle):
    """Retrieve TX and RX throughput for a single GPU."""
    tx_throughput = pynvml.nvmlDeviceGetPcieThroughput(handle, pynvml.NVML_PCIE_UTIL_TX_BYTES)
    rx_throughput = pynvml.nvmlDeviceGetPcieThroughput(handle, pynvml.NVML_PCIE_UTIL_RX_BYTES)
    return tx_throughput, rx_throughput

def monitor_gpu_pcie_to_dataframe():
    """Monitor PCIe throughput for all GPUs and store the results in a DataFrame."""
    # Initialize NVML
    pynvml.nvmlInit()
    df = pd.DataFrame()  # Initialize an empty DataFrame
    data = {
        'time': [],
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
        count_down = 100
        print("\t".join(data.keys()))
        
        while count_down > 0:
            count_down -= 1
            # get tx (data from gpu to host) and rx (data from host to gpu)
            # for each gpu
            t = time.time()
            tx0, rx0 = get_pcie_throughput(handle0)
            data['time'].append(t)
            data['gpu0-from'].append(tx0)
            data['gpu0-to'].append(rx0)
            tx1, rx1 = get_pcie_throughput(handle1)
            data['gpu1-from'].append(tx1)
            data['gpu1-to'].append(rx1)
            tx2, rx2 = get_pcie_throughput(handle2)
            data['gpu2-from'].append(tx2)
            data['gpu2-to'].append(rx2)
            tx3, rx3 = get_pcie_throughput(handle3)
            data['gpu3-from'].append(tx3)
            data['gpu3-to'].append(rx3)
            
            print("\t".join(map(str, [f"{t:.2f}", tx0, rx0, tx1, rx1, tx2, rx2, tx3, rx3])))
            
            
            if len(data['time']) > 20 and tx0 <= 50 and rx0 <= 50 and tx1 <= 50 and rx1 <= 50 and tx2 <= 50 and rx2 <= 50 and tx3 <= 50 and rx3 <= 50:
                count_down = 5
 
    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        # Shutdown NVML
        pynvml.nvmlShutdown()
       
        df = pd.DataFrame(data)
    return df

if __name__ == "__main__":
    # Monitor PCIe throughput and save to a DataFrame
    data = monitor_gpu_pcie_to_dataframe()

    # Optionally, save to a CSV file
    data.to_csv("gpu_pcie_throughput.csv", index=False)
