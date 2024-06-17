# General Checkup 

import GPUtil
import psutil

import torch

import pytorch_lightning as pl

# check pytorch_lightning
print('Versione: ',pl.__version__)

# Check CPU Memory Usage


cpu_memory = psutil.virtual_memory()
print(f"Total CPU Memory: {cpu_memory.total / (1024 ** 3):.2f} GB")
print(f"Available CPU Memory: {cpu_memory.available / (1024 ** 3):.2f} GB")
print(f"Used CPU Memory: {cpu_memory.used / (1024 ** 3):.2f} GB")

try:
    

    gpus = GPUtil.getGPUs()
    for i, gpu in enumerate(gpus):
        print(f"GPU {i + 1}:")
        print(f"    GPU Name: {gpu.name}")
        print(f"    Total GPU Memory: {gpu.memoryTotal} MB")
        print(f"    Used GPU Memory: {gpu.memoryUsed} MB")
        print(f"    Free GPU Memory: {gpu.memoryFree} MB")
except ImportError:
    print("GPUtil is not installed. Cannot check GPU memory usage.")


if torch.cuda.is_available():
    print("GPU is available")
else:
    print("GPU is not available")





#################################################################################################################

