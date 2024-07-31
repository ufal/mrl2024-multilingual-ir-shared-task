from transformers import AutoModelForCausalLM, AutoTokenizer
# from transformers.models.llama.modeling_llama import LlamaModel

not_sharded_path = "/home/manea/personal_work_ms/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/not_sharded/llama-3-8B"
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", use_auth_token=True, device_map='auto') #, local_files_only=True)

model.save_pretrained(not_sharded_path, safe_serialization=False)