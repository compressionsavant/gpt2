# https://huggingface.co/datasets/compressionsavant/fw-edu/tree/main
import os
import argparse
import tiktoken
import numpy as np
import multiprocessing as mp
from datasets import load_dataset
from huggingface_hub import HfApi
from tqdm import tqdm

SHARD_HEADER = {
    "magic": 716202,
    "version": 1,
    "dtype": np.uint16
}

def tokenize(doc):
    enc = tiktoken.get_encoding("gpt2")
    eot = enc._special_tokens["<|endoftext|>"]
    tokens = [eot]
    tokens.extend(enc.encode_ordinary(doc["text"]))
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token dictionary too large for uint16"
    tokens_np = tokens_np.astype(np.uint16)
    return tokens_np

def write_shard(filename, tokens):
    header = np.empty(256, dtype=np.int32)
    header[0] = SHARD_HEADER["magic"]
    header[1] = SHARD_HEADER["version"]
    header[2] = len(tokens)
    tokens = np.array(tokens, dtype=SHARD_HEADER["dtype"])

    with open(filename, "wb") as f:
        f.write(header.tobytes())
        f.write(tokens.tobytes())

def main():
    parser = argparse.ArgumentParser(description="generate fineweb shards")
    parser.add_argument('--token_size', type=int, required=True, help='size of fineweb tokens to sample')
    parser.add_argument('--path', type=str, required=True, help='path to external drive')
    parser.add_argument('--repo', type=str, default="compressionsavant/fw-edu", help='repo to store the fineweb shards')
    args = parser.parse_args()

    assert os.path.isdir(args.path)
    assert args.token_size in [10, 100, 350] # token size

    fineweb_dir = os.path.join(args.path, f"edu_fineweb{args.token_size}B")
    os.makedirs(fineweb_dir, exist_ok=True)
    fw = load_dataset("HuggingFaceFW/fineweb-edu", name=f"sample-{args.token_size}BT", split="train", cache_dir=fineweb_dir)


    shard_size = int(1e8)
    num_shards = int(args.token_size * 1e9) // shard_size
    nprocs = max(1, os.cpu_count() // 2)
    with mp.Pool(nprocs) as pool:
        shard_idx = token_cnt = 0
        tokens = np.empty((shard_size,), dtype=np.uint16)
        progress_bar = tqdm(total=shard_size, unit="token", desc=f"tokenizing fineweb shard: {shard_idx} of {num_shards - 1}")
    
        for batch in pool.imap(tokenize, fw, chunksize=16):
            if token_cnt + len(batch) > shard_size:
                split = "val" if shard_idx == 0 else "train"
                filename = os.path.join(fineweb_dir, f"edu_fineweb{args.token_size}B_{split}_{shard_idx:06d}.bin")
                remainder = shard_size - token_cnt
                tokens[token_cnt : token_cnt + remainder] = batch[:remainder]
                write_shard(filename, tokens.tolist())
                progress_bar.update(remainder)
                shard_idx += 1
                progress_bar = None
                tokens[0 : len(batch) - remainder] = batch[remainder:]
                token_cnt = len(batch) - remainder

            else:
                tokens[token_cnt : token_cnt + len(batch)] = batch
                token_cnt += len(batch)

                if progress_bar is None:
                    progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"tokenizing fineweb shard: {shard_idx} of {num_shards - 1}")
                progress_bar.update(len(batch))

    
        if token_cnt > 0:
            split = "val" if shard_idx == 0 else "train"
            filename = os.path.join(fineweb_dir, f"edu_fineweb{args.token_size}B_{split}_{shard_idx:06d}.bin")
            write_shard(filename, tokens[:token_cnt].tolist())
        
        fineweb_files = [file for file in os.listdir(fineweb_dir) if file.endswith(".bin")]
        assert len(fineweb_files) == num_shards, "number of shards does not match"

        api = HfApi()
        api.upload_large_folder(repo_id=args.repo,
                                repo_type="dataset", 
                                folder_path=fineweb_dir, 
                                num_workers=nprocs, 
                                allow_patterns="*.bin",
                                ignore_patterns="*.bin.lock")
        
if __name__ == "__main__":
    main()
