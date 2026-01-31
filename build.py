import re
import torch
from huggingface_hub import list_repo_tree, hf_hub_download
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer

def get_last_checkpoint(repo):
    tree = list_repo_tree(repo, repo_type="model")    
    models = [file.path + "/checkpoint.pth" for file in tree if file.path.startswith("step")]
    models.sort(key=lambda x: int(re.search(r'\d+', x).group()))
    return models[-1]

def convert_checkpoint(file):
    checkpoint_path = hf_hub_download(repo_id="compressionsavant/gpt2", filename=file)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model"]
    state_dict = {k.lstrip("_orig_mod."): v for k,v in state_dict.items()}
    # slice and re-tie the weights
    sliced_lm_head = state_dict["lm_head.weight"][:50257]
    state_dict["lm_head.weight"] = sliced_lm_head
    state_dict["transformer.wte.weight"] = sliced_lm_head
    assert state_dict["lm_head.weight"] is state_dict["transformer.wte.weight"]
    for k in state_dict:
        if any(suffix in k for suffix in ["c_attn", "c_proj", "c_fc"]):
            if k.endswith(".weight"):
                state_dict[k] = state_dict[k].t()
    
    return state_dict

if __name__ == '__main__':
    checkpoint = get_last_checkpoint("compressionsavant/gpt2")
    sd = convert_checkpoint(checkpoint)

    config = GPT2Config(
            vocab_size=50257,
            n_positions=1024,
            n_embd=768,
            n_layer=12,
            n_head=12,
            activation_function="gelu",
            architectures=["GPT2LMHeadModel"],
            model_type="gpt2",
            initializer_range=0.02,
            layer_norm_epsilon=1e-05,
            attn_pdrop=0.1,
            resid_pdrop=0.1,
            embd_pdrop=0.1,
            bos_token_id=50256,
            eos_token_id=50256,
        )

    model = GPT2LMHeadModel(config)
    model.load_state_dict(sd)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    model.push_to_hub("compressionsavant/gpt2-124M")
    tokenizer.push_to_hub("compressionsavant/gpt2-124M")
