# GPT-2
A minimal GPT-2(124M) reproduction with GPT-3 Hyperparams trained on 10B token of [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)

## Architecture
Following GPT-2 Architecture with extended vocab_size(50257 -> 50304) for training efficency. Later sliced back for compability

## Training
Trained with DDP on 8xH100s

| Parameter | Value |
|---|---|
| Batch size | 524,288 tokens |
| Micro batch | 64 |
| Sequence length | 1024 |
| Learning rate | 6e-4 (cosine → 6e-5) |
| Warmup | 375M tokens |
| Weight decay | 0.1 |
| Optimizer | AdamW (0.9, 0.95) |
| Precision | bfloat16 |
| Total tokens | 10B (1 epoch) |

```bash
torchrun --standalone --nproc_per_node=8 train.py
```

Checkpoints are saved to [compressionsavant/gpt2](https://huggingface.co/compressionsavant/gpt2) on Hugging Face every 5000 steps.

## Data
 
Tokenized shards are stored in a custom binary format (very similar to llm.c). To generate shards from FineWeb-Edu:
 
```bash
python data/fineweb_edu.py --token_size 10 --path /path/to/storage
```

## Evaluation
Hellaswag and MMLU(zero-shot) use average cross-entropy over answer tokens:

```bash
PYTHONPATH="." uv run eval/hellaswag.py
PYTHONPATH="." uv run eval/mmlu.py
```

## Export
 
Export last Checkpoint to GPT2LMHeadModel format:
 
```bash
PYTHONPATH="." uv build.py
```

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model = GPT2LMHeadModel.from_pretrained("compressionsavant/gpt2-124M")
tokenizer = GPT2Tokenizer.from_pretrained("compressionsavant/gpt2-124M")
model.eval()

prompt = "The meaning of life is"
input_ids = tokenizer.encode(prompt, return_tensors="pt")

output = model.generate(
    input_ids,
    max_new_tokens=100,
    do_sample=True,
    top_k=50,
    top_p=0.95,
    temperature=0.8,
)

print(tokenizer.decode(output[0]))
```



