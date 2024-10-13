import numba.cuda

device = numba.cuda.get_current_device() # CUDA enabled by GPU is assigned to device
#print attributes of cuda
free_memory, total_memory =numba.cuda.current_context().get_memory_info()
print(f"Device name: {device.name}")
print(f"Total memory: {total_memory/(1024**3):.2f}GB")
print(f"Multiprocessor Count: {device.MULTIPROCESSOR_COUNT}")
