from transformers import AutoModelForCausalLM
from __init__ import llama3_not_sharded_path, llama3_original_path


model = AutoModelForCausalLM.from_pretrained(llama3_original_path, use_auth_token=True, device_map='auto')
model.save_pretrained(llama3_not_sharded_path, safe_serialization=False)