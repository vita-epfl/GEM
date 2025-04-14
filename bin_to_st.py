import os

import torch
from safetensors.torch import save_file
import sys


ckpt = #checkpoints
vista_bin = torch.load(ckpt, map_location="cpu")  # only contains model weights

print(len(vista_bin.keys()))

for k in list(vista_bin.keys()):  # merge LoRA weights (if exist) for inference
    if "adapter_down" in k:
        print("adapter")
        if "q_adapter_down" in k:
            up_k = k.replace("q_adapter_down", "q_adapter_up")
            pretrain_k = k.replace("q_adapter_down", "to_q")
        elif "k_adapter_down" in k:
            up_k = k.replace("k_adapter_down", "k_adapter_up")
            pretrain_k = k.replace("k_adapter_down", "to_k")
        elif "v_adapter_down" in k:
            up_k = k.replace("v_adapter_down", "v_adapter_up")
            pretrain_k = k.replace("v_adapter_down", "to_v")
        else:
            up_k = k.replace("out_adapter_down", "out_adapter_up")
            if "model_ema" in k:
                pretrain_k = k.replace("out_adapter_down", "to_out0")
            else:
                pretrain_k = k.replace("out_adapter_down", "to_out.0")

        lora_weights = vista_bin[up_k] @ vista_bin[k]
        del vista_bin[k]
        del vista_bin[up_k]
        vista_bin[pretrain_k] = vista_bin[pretrain_k] + lora_weights

print(len(vista_bin.keys()))

for k in list(vista_bin.keys()):  # remove the prefix
    if "_forward_module" in k and "decay" not in k and "num_updates" not in k:
        print("forward")
        vista_bin[k.replace("_forward_module.", "")] = vista_bin[k]
        del vista_bin[k]

print(len(vista_bin.keys()))

print("model_ema.diffusion_modeltime_embed0weight" in vista_bin.keys())

vista_st = dict()
for k in list(vista_bin.keys()):
    vista_st[k] = vista_bin[k]


os.makedirs("ckpts", exist_ok=True)

save_file(vista_st, "ckpts/gem.safetensors")
