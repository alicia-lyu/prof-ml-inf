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
        'gpu0_to': [],
        'gpu0_from': [],
        'gpu1_to': [],
        'gpu1_from': [],
        'gpu2_to': [],
        'gpu2_from': [],
        'gpu3_to': [],
        'gpu3_from': [],
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
            data['gpu0_from'].append(tx0)
            data['gpu0_to'].append(rx0)
            tx1, rx1 = get_pcie_throughput(handle1)
            data['gpu1_from'].append(tx1)
            data['gpu1_to'].append(rx1)
            tx2, rx2 = get_pcie_throughput(handle2)
            data['gpu2_from'].append(tx2)
            data['gpu2_to'].append(rx2)
            tx3, rx3 = get_pcie_throughput(handle3)
            data['gpu3_from'].append(tx3)
            data['gpu3_to'].append(rx3)
            
            print("\t".join(map(str, [f"{t:.2f}", rx0, tx0, rx1,tx1, rx2, tx2, rx3, tx3])))
            
            
            if count_down > 5 and len(data['time']) > 20 and tx0 <= 150 and rx0 <= 150 and tx1 <= 150 and rx1 <= 150 and tx2 <= 150 and rx2 <= 150 and tx3 <= 150 and rx3 <= 150:
                count_down = 5
 
    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        # Shutdown NVML
        pynvml.nvmlShutdown()
       
        df = pd.DataFrame(data)
        
    return df

if __name__ == "__main__":
    data = monitor_gpu_pcie_to_dataframe()
    data.to_csv("gpu_pcie_throughput.csv", index=False)
