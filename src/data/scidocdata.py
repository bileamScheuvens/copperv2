from itertools import chain

import lightning as L
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset, random_split

from src.constants import DATAPATH, QUERY_CLASSES
from src.data.load_data import ebm_iter, pubmed_iter
from src.data.sampling import SamplerFactory
from src.utils.configs import SciDocDataConfig


class ExtractionsDatamodule(L.LightningDataModule):
    def __init__(
        self,
        cfg: SciDocDataConfig,
        tokenizer,
    ):
        super().__init__()
        self.batch_size = cfg.batch_size
        self.cfg = cfg
        self.shared_dim = 128  # TODO make configurable
        self.tokenizer = tokenizer

    def setup(self, stage: str):
        dataset = load_dataset(
            "parquet",
            data_files=str(DATAPATH / "index-pubmed" / "extractions.parquet"),
            split="train",
        )
        split = dataset.train_test_split(
            train_size=self.cfg.train_ratio, seed=self.cfg.seed
        )
        self.train_data, self.val_data = split["train"], split["test"]

    def train_dataloader(self):
        sampler = SamplerFactory(self.cfg, self.train_data)
        return DataLoader(
            self.train_data, batch_sampler=sampler, collate_fn=self.collate
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data, batch_size=self.batch_size, collate_fn=self.collate
        )

    def collate(self, rows):
        docs = self.tokenizer(
            [f"{r['title']}\n\n{r['abstract']}" for r in rows],
            padding=True,
            truncation=True,
            return_tensors="pt",
        )["input_ids"]

        # dummy extractions TODO replace with lookups from some cache
        max_elements = 0
        for r in rows:
            n_elements = 0
            for c in QUERY_CLASSES:
                n_elements += len(r[c] or "")
            max_elements = max(max_elements, n_elements)
        B = len(rows)
        N = max_elements
        S = self.shared_dim
        queries = torch.zeros(B, N, S)
        classes = torch.zeros(B, S)
        return docs, queries, classes


class SciDocDataset(Dataset):
    """Dataset that yields title-abstract pairs. Wraps ebm_nlp and/or dump of pubmed RCTs. ebm_nlp contains all but 541/4993 duplicates."""

    def __init__(self, cfg: SciDocDataConfig):
        super().__init__()
        self.cfg = cfg

        self.pmids = []
        self.titles = {}
        self.abstracts = {}
        self.pmid2idx = {}

        corpus_iter = []
        if cfg.use_ebm_nlp:
            data_iter, _ = ebm_iter()
            corpus_iter = chain(corpus_iter, data_iter)
        if cfg.use_pubmed_rct:
            data_iter, _ = pubmed_iter()
            corpus_iter = chain(corpus_iter, data_iter)

        for i, (pmid, title, abstract) in enumerate(corpus_iter):
            if pmid in self.titles:
                continue
            self.pmids.append(pmid)
            self.titles[pmid] = title
            self.abstracts[pmid] = abstract
            self.pmid2idx[pmid] = i

    def __len__(self):
        return len(self.pmids)

    def __getitem__(self, idx):  # ty:ignore[invalid-method-override]
        pmid = self.pmids[idx]
        return self.titles[pmid], self.abstracts[pmid]

    def get_by_pmid(self, pmid):
        return self.titles[pmid], self.abstracts[pmid]


class SciDocDatamodule(L.LightningDataModule):
    def __init__(self, cfg: SciDocDataConfig):
        super().__init__()
        self.batch_size = cfg.batch_size
        self.cfg = cfg

    def setup(self, stage: str):
        if stage == "fit":
            self.train_data, self.val_data = random_split(
                dataset=SciDocDataset(self.cfg),
                lengths=[self.cfg.train_ratio, self.cfg.val_ratio],
                generator=torch.Generator().manual_seed(self.cfg.seed),
            )

    def train_dataloader(self):
        sampler = SamplerFactory(self.cfg, self.train_data)
        return DataLoader(self.train_data, num_workers=0, batch_sampler=sampler)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, num_workers=0)
