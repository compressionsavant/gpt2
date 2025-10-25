# pulls shards for training run
import os
from huggingface_hub import hf_hub_download

def download(filename):
    fineweb_dir = os.path.join(os.path.dirname(__file__), "edu_fineweb10B")
    os.makedirs(fineweb_dir, exist_ok=True)
    if not os.path.exists(os.path.join(fineweb_dir, filename)):
        hf_hub_download(repo_id="compressionsavant/fw-edu",
                        filename=filename,
                        repo_type="dataset",
                        local_dir=fineweb_dir)

num_shards = 100
for i in range(num_shards):
    if i == 0:
        download("edu_fineweb10B_val_%06d.bin" % i)
    else:
        download("edu_fineweb10B_train_%06d.bin" % i)

