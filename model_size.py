import re
from collections import defaultdict

import torch
from transformers import AutoModelForCausalLM

model_path = "/shared/public/elr-models/Qwen/Qwen3-30B-A3B/67b0e0ca24de1b8cedea4c97f1925df66d72bee1"

model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
)
print(model)

# Dictionary to store parameter counts per submodule
param_groups = defaultdict(int)

# Traverse all named parameters
for name, param in model.named_parameters():
    if not param.requires_grad:
        continue
    if "embed_tokens" in name:
        param_groups["Embedding"] += param.numel()
    elif "lm_head" in name:
        param_groups["LM Head"] += param.numel()
    elif ".self_attn.q_proj" in name:
        param_groups["Attention/q_proj"] += param.numel()
    elif ".self_attn.k_proj" in name:
        param_groups["Attention/k_proj"] += param.numel()
    elif ".self_attn.v_proj" in name:
        param_groups["Attention/v_proj"] += param.numel()
    elif ".self_attn.o_proj" in name:
        param_groups["Attention/o_proj"] += param.numel()
    elif ".self_attn.q_norm" in name:
        param_groups["Attention/q_norm"] += param.numel()
    elif ".self_attn.k_norm" in name:
        param_groups["Attention/k_norm"] += param.numel()
    elif ".mlp.gate_proj" in name:
        param_groups["MLP/gate_proj"] += param.numel()
    elif ".mlp.up_proj" in name:
        param_groups["MLP/up_proj"] += param.numel()
    elif ".mlp.down_proj" in name:
        param_groups["MLP/down_proj"] += param.numel()
    elif ".mlp.gate" in name:
        param_groups["MLP/gate"] += param.numel()
    elif re.search(r"\.experts\.\d+\.gate_proj", name):
        param_groups["MLP/experts/gate_proj"] += param.numel()
    elif re.search(r"\.experts\.\d+\.up_proj", name):
        param_groups["MLP/experts/up_proj"] += param.numel()
    elif re.search(r"\.experts\.\d+\.down_proj", name):
        param_groups["MLP/experts/down_proj"] += param.numel()
    elif "input_layernorm" in name:
        param_groups["LayerNorm/input"] += param.numel()
    elif "post_attention_layernorm" in name:
        param_groups["LayerNorm/post_attention"] += param.numel()
    elif name.startswith("model.norm"):
        param_groups["LayerNorm/final"] += param.numel()
    elif "rotary_emb" in name:
        param_groups["RotaryEmbedding"] += param.numel()
    else:
        param_groups["Other"] += param.numel()

total_params = sum(p.numel() for p in model.parameters())
print(f"Total params: {total_params / 1e9:.2f}B")

total_params_calculated = sum(v for k, v in param_groups.items() if k != "Other")
print(f"Total params calculated: {total_params_calculated / 1e9:.2f}B")
