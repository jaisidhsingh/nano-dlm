import json

import matplotlib.pyplot as plt

with open("/fast/jsingh/nano-dlm_single_gpu_42M_test_run.json") as f:
    data = json.load(f)

x = data["train"].keys()
train_ppl = [data["train"][k]["ppl"] for k in x]
val_ppl = [data["val"][k]["ppl"] for k in x]

plt.plot(x, train_ppl, label="train")
plt.plot(x, val_ppl, label="val")
plt.legend()
plt.ylabel("Perplexity")
plt.xlabel("Steps")
plt.savefig("./test.png")
