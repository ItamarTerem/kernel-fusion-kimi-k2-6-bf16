from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from kimi_patch import patch_kimi_model

model_id = "moonshotai/Kimi-K2.6"   # or your local checkpoint path

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

# Apply the fusion — choose "V2" for DGX Spark / B200, "V3" for H100
model = patch_kimi_model(model, variant="V2")

# Quick sanity check
inputs = tokenizer("Hello, world!", return_tensors="pt").to(model.device)
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=20)
print(tokenizer.decode(out[0], skip_special_tokens=True))
