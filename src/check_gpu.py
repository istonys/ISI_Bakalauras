import os
import torch

print("Python PID:", os.getpid())
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())

if torch.cuda.is_available():
    print("Current CUDA device:", torch.cuda.current_device())
    print("Device name:", torch.cuda.get_device_name(0))
    x = torch.randn(1024, 1024, device="cuda")
    y = x @ x
    print("CUDA tensor test OK:", float(y.mean().detach().cpu()))
else:
    print("CUDA neveikia. Mokymas vyks CPU, bet eksperimentams tai netinkama.")
