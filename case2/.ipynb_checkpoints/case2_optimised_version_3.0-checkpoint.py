# Version 3.0

import random
import yaml
import time
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from helpers import mG


class InfiniteDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, "batch_sampler", _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler:
    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


def preprocess(text, text_filter):
    text = "".join(
        c if c in text_filter else "" for c in str(text)
    )  # remove characters not in filter
    text = text.lower().strip().replace(",", "").replace("\n", "").replace(" ", "")
    return text


class ClassifierDataset(Dataset):
    def __init__(self, text, cfg):
        super().__init__()
        self.text = text
        self.cfg = cfg
        self.TEXT_FILTER = (
            "1234567890:;&'()+-./, ABCDEFGHIJKLMNOPQRSTUVWXYZẞabcdefghijklmnopqrstuvwxyzß"
        )
        vocab_tokens = ["<PAD>", "<BEG>", "<END>", "<UNK>"] + list(self.TEXT_FILTER)
        self.vocab = {
            "itos": {idx: value for idx, value in enumerate(vocab_tokens)},
            "stoi": {value: idx for idx, value in enumerate(vocab_tokens)},
        }

        print(f"Creating dataset with a length of {self.cfg['datasetSize']}")
        ts = time.time()
        # clean
        self.text = self.preprocess(self.text)
        # augment & generate
        self.src = [
            mG(self.text, self.cfg["maxModifications"])
            for _ in tqdm(range(self.cfg["datasetSize"]), desc="Augmenting")
        ]
        self.label = [random.randint(0, 1) for _ in range(self.cfg["datasetSize"])]
        # tokenize
        for i in tqdm(range(len(self.src)), desc="Tokenizing"):
            self.src[i], self.label[i] = self.tokenize(
                (self.src[i], self.label[i]), self.vocab, self.cfg
            )
        te = time.time()
        time_taken = te - ts
        print(f"Time taken to process dataset: {time_taken} seconds")
        print("Dataset Processed!")

    def preprocess(self, text):
        text = "".join(c if c in self.TEXT_FILTER else "" for c in str(text))
        text = text.lower().strip().replace(",", "").replace("\n", "")
        return text.lower()

    @staticmethod
    def tokenize(batch, tokenizer, cfg):
        src, label = batch
        src = ["<BEG>"] + list(src) + ["<END>"]
        src = src + ["<PAD>"] * (cfg["maxLetterCount"] - len(src))
        src_tokens = [
            tokenizer.get("stoi").get(item, tokenizer.get("stoi").get("<UNK>"))
            for item in src
        ]
        src_tokens = src_tokens[: cfg["maxLetterCount"]]
        
        # Ensure src_tokens has the expected structure and type
        if isinstance(src_tokens, list):
            src_tokens_length = len(src_tokens)
        else:
            src_tokens_length = 0
        
        return src_tokens, label


    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        src = self.src[idx]
        label = self.label[idx]

        return torch.tensor(src, dtype=torch.float32), torch.tensor(label, dtype=torch.float32).view(-1)

    @staticmethod
    def collate_fn(batch):
        src, label = zip(*batch)
        return torch.stack(src, 0), torch.stack(label, 0)

    def __repr__(self) -> str:
        return f"ClassName: {__class__.__name__}, SourceDataLength: {len(self)}"


def create_dataloader(text, cfg):
    dataset = ClassifierDataset(text, cfg)
    tokenizer = dataset.vocab
    return (
        dataset,
        InfiniteDataLoader(
            dataset,
            batch_size=cfg["batchSize"],
            num_workers=cfg["workers"],
            sampler=None,
            pin_memory=True,
            collate_fn=ClassifierDataset.collate_fn,
        ),
        tokenizer,
    )

import numpy as np

def tokenize(batch, tokenizer, cfg):
    src, label = batch

    src_tokens = np.array(src)  # Convert to numpy array
    max_len = cfg["maxLetterCount"]

    # Pad src_tokens with <PAD> token
    pad_width = ((0, 0), (0, max_len - len(src_tokens)))
    src_tokens = np.pad(src_tokens, pad_width, constant_values=tokenizer['stoi']['<PAD>'])

    return src_tokens.tolist(), label


if __name__ == "__main__":
    CFG_PATH = "./cfg.yaml"
    with open(CFG_PATH, "r") as f:  # load cfg
        cfg = yaml.safe_load(f)

    text = "01:ccInterview-02:ccInterview-03:ccInterview-04:ccInterview"

    dataset, dataloader, tokenizer = create_dataloader(text, cfg)

    print(''.join([tokenizer.get("itos").get(int(item), "<UNK>") for item in next(iter(dataloader))[0][0].flatten().numpy().tolist()]))

    del dataset, dataloader, tokenizer
