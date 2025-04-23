import torch
assert torch.cuda.is_available(), "CUDA is not available. Please check your GPU setup."
print("Using GPU: ", torch.cuda.get_device_name(0))app