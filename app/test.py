import torch
import torchvision

print(torch.__version__)
print(torchvision.__version__)

print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("❌ GPU not available - check install")
