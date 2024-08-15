import matplotlib.pyplot as plt
import json
import ipdb

file_path = "slurm_logs/AndreiLLM_job.4775624.out"

with open(file_path, "r") as f:
    lines = f.read().split("\n")[7:-1]

logs = [json.loads(line.replace("'", '"')) for line in lines]

gathered = {}
for log_dict in logs:
    for key, value in log_dict.items():
        if key not in gathered:
            gathered[key] = []
        
        gathered[key].append(value)


plt.figure(figsize=(10, 5))
for key in ["loss", "eval_loss"]:
    values = gathered[key]
    plt.plot(values, label=key)

plt.savefig("losses.png")