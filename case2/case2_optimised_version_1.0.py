# Version 1.0
import yaml
import time
import torch
import random
from tqdm import tqdm
from helpers import mG
# from multiprocessing import Pool
from torch.utils.data import Dataset, DataLoader
from concurrent.futures import ProcessPoolExecutor

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


def tokenize(args):
    src, label, tokenizer, cfg = args
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
        start_time = time.time()
        with ProcessPoolExecutor() as executor:
            self.src = list(tqdm(executor.map(mG, [(self.text, self.cfg["maxModifications"])]*self.cfg["datasetSize"]), total=self.cfg["datasetSize"], desc="Augmenting"))
        print(f"Augmenting took {time.time() - start_time} seconds")
        self.label = [random.randint(0, 1) for _ in range(self.cfg["datasetSize"])]
        # tokenize
        start_time = time.time()
        with ProcessPoolExecutor() as executor:
            self.src, self.label = zip(*tqdm(executor.map(tokenize, [(src, label, self.vocab, self.cfg) for src, label in zip(self.src, self.label)]), total=len(self.src), desc="Tokenizing"))
        print(f"Tokenizing took {time.time() - start_time} seconds")


        
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

    for batch in dataloader:
        src, label = batch
        print(''.join([tokenizer.get("itos").get(int(item), "<UNK>") for item in src[0].flatten().tolist()]))
        break

    del dataset, dataloader, tokenizer