from transformers import AutoModelForSeq2SeqLM
from __init__ import aya_model_name, aya_model_not_sharded_path


model = AutoModelForSeq2SeqLM.from_pretrained(aya_model_name, use_auth_token=True, device_map='auto')
model.save_pretrained(aya_model_not_sharded_path, safe_serialization=False)