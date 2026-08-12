import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch

print(torch.__version__)
print(torch.backends.mps.is_available())  # should be True on Apple Silicon (M-series)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Device:", device)