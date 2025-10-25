# create a hellaswag dataset
import os
import json
import requests
from tqdm import tqdm
from datasets import Dataset, Features, Value, Sequence, DatasetDict


hellaswag_remote = {
    "train": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_train.jsonl",
    "val": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl",
    "test": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_test.jsonl",
}


class HellaswagDataset:
    def __init__(self):
        self.dir = os.path.join(os.path.dirname(__file__), "hellaswag")
        self.remote = hellaswag_remote
        self.features = Features({
            "ind": Value("int32"),
            "activity_label": Value("string"),
            "ctx_a": Value("string"),
            "ctx_b": Value("string"),
            "ctx": Value("string"),
            "split": Value("string"),
            "split_type": Value("string"),
            "label": Value("int32"),
            "endings": Sequence(Value("string")),
            "source_id": Value("string")
        })


    def download(self, split):
        assert split in self.remote
        url = self.remote[split]
        os.makedirs(self.dir, exist_ok=True)
        hs_file = os.path.join(self.dir, f"hellaswag_{split}.jsonl")
        if not os.path.exists(hs_file):
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0))
                fname = os.path.basename(url)
                with (
                    open(hs_file, "wb") as f,
                    tqdm(desc=fname, total=total, unit="B", 
                         unit_scale=True) as bar
                ):
                    for chunk in r.iter_content(chunk_size=512):
                        sz = f.write(chunk)
                        bar.update(sz)
        else:
            print(f"File {hs_file} already exists")
        
        return hs_file


    def create_dataset(self, split):
        assert split in self.remote
        hs_file = self.download(split)
        with open(hs_file, "r") as f:
            data = [json.loads(line) for line in f]
        if "label" not in data[0]:
            for item in data:
                item["label"] = -1
        dataset = Dataset.from_list(data, features=self.features)
        return dataset
    
    @classmethod
    def create_all_datasets(cls):
        instance = cls()
        datasets = {}
        for split in cls.remote.keys():
            datasets[split] = instance.create_dataset(split)
        return DatasetDict(datasets)


if __name__ == "__main__":
    datasets = HellaswagDataset.create_all_datasets()
    datasets.push_to_hub(repo_id="compressionsavant/hellaswag")

