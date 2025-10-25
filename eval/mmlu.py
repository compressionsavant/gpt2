import os
import torch
import torch.nn.functional as F
import tiktoken
from tqdm import tqdm
from datasets import load_dataset
from huggingface_hub import list_repo_tree, hf_hub_download
from model import GPT

enc = tiktoken.get_encoding("gpt2")
device = "cuda" if torch.cuda.is_available() else "cpu"

def iter_checkpoints(repo):
    tree = list_repo_tree(repo, repo_type="model")    
    models = [file.path + "/checkpoint.pth" for file in tree if file.path.startswith("step")]
    return models

def load_checkpoint(file):
    checkpoint_path = hf_hub_download(repo_id="compressionsavant/gpt2", filename=file)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_config = checkpoint["config"]
    model_data = checkpoint["model"]
    model_data = {k.lstrip("_orig_mod."): v for k,v in model_data.items()}
    model = GPT(model_config)
    model.load_state_dict(model_data)
    model.to(device)
    return model

def render_completion(ctx, endings):
    _ctx = enc.encode("Question: " + ctx + "\n\nAnswer:")
    len_ctx = len(_ctx)
    mask = []
    tokt = [torch.tensor(_ctx + enc.encode(" " + ending)) for ending in endings]
    t_len = max(len(t) for t in tokt)
    prompt = torch.zeros((len(endings), t_len), dtype=torch.long)
    for i, tokens in enumerate(tokt):
        prompt[i, :len(tokens)] = tokens
        mask.append(len(tokens))
    prompt = prompt.to(device)
    return len_ctx, mask, prompt

@torch.no_grad()
def evaluate(model_list):
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    for model_path in model_list:
        n_total = n_correct = 0
        acc = 0.0

        model = load_checkpoint(model_path)
        model.eval()
        dataset = load_dataset("cais/mmlu", "all", split="test").shuffle(seed=42)
        bar = tqdm(dataset, desc=f"Evaluating {os.path.dirname(model_path)}")
        for example in bar:
            ctx = example["question"]
            endings = example["choices"]
            label = example["answer"]

            len_ctx, mask, tokens = render_completion(ctx, endings)
            logits, _ = model(tokens, return_logits=True)
            shifted_logits = logits[:, :-1, :].contiguous()
            shifted_tokens = tokens[:, 1:].contiguous()
            flat_shifted_logits = shifted_logits.view(-1, shifted_logits.size(-1))
            flat_shifted_tokens = shifted_tokens.view(-1)
        
            losses = F.cross_entropy(flat_shifted_logits, flat_shifted_tokens, reduction="none")
            losses = losses.view(tokens.size(0), -1)
            # slice-out unwanted losses
            completion_losses = []
            for i, completion in enumerate(mask):
                completion_loss = losses[i][len_ctx-1 : completion-1]
                avg_loss = completion_loss.mean().item()
                completion_losses.append(avg_loss)

            pred = torch.tensor(completion_losses).argmin().item()
            if label == pred:
                n_correct += 1
            n_total += 1

            acc = n_correct / n_total
            bar.set_postfix({"Accuracy": f"{acc:.4f}"})


if __name__ == "__main__":
    model_list = iter_checkpoints("compressionsavant/gpt2")
    evaluate(model_list)

