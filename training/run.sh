#!/bin/bash
pip install -r requirements.txt

python3 -c "
from huggingface_hub import login
login(token='$HF_TOKEN')
"

!rm -rf dataset model_binaries
torchrun --standalone --nproc_per_node=2 main.py
