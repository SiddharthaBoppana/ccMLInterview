import random
import yaml
import time
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
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


def tokenize(batch, tokenizer, cfg):

    src, label = batch

    src = ["<BEG>"] + list(src) + ["<END>"]
    src = src + ["<PAD>"] * (cfg["maxLetterCount"] - len(src))
    src = [
        tokenizer.get("stoi").get(item, tokenizer.get("stoi").get("<UNK>"))
        for item in src
    ]
    src = src[: cfg["maxLetterCount"]]
    return src, label


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

        def preprocess(text):
            text = "".join(
                c if c in self.TEXT_FILTER else "" for c in str(text)
            )  # remove characters not in filter
            text = text.lower()
            text = text.strip()
            text = text.replace(",", "")
            text = text.replace("\n", "")
            text = text.lower()
            return text

        print(f"Creating dataset with a length of {self.cfg['datasetSize']}")
        ts = time.time()
        # clean
        self.text = preprocess(self.text)
        # augment & generate
        self.src = [
            mG(self.text, self.cfg["maxModifications"])
            for _ in tqdm(range(self.cfg["datasetSize"]), desc="Augmenting")
        ]
        self.label = [random.randint(0, 1) for _ in range(self.cfg["datasetSize"])]
        # tokenize
        for i in tqdm(range(len(self.src)), desc="Tokenizing"):
            self.src[i], self.label[i] = tokenize(
                (self.src[i], self.label[i]), self.vocab, self.cfg
            )
        te = time.time()
        time_taken = te - ts
        print(f"Time taken to process dataset: {time_taken} seconds")
        print("Dataset Processed!")
        
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


if __name__ == "__main__":

    CFG_PATH = "./cfg.yaml"
    with open(CFG_PATH, "r") as f:  # load cfg
        cfg = yaml.safe_load(f)

    text = "01:ccInterview-02:ccInterview-03:ccInterview-04:ccInterview"

    dataset, dataloader, tokenizer = create_dataloader(text, cfg)

    print(''.join([tokenizer.get("itos").get(int(item), "<UNK>") for item in next(iter(dataloader))[0][0].flatten().numpy().tolist()]))
    
    del dataset, dataloader, tokenizer

