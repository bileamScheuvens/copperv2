from src.utils.configs import SciDocDataConfig
from itertools import chain

import lightning as L
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset, random_split
from torch.nn.utils.rnn import pad_sequence

from src.constants import DATAPATH, QUERY_CLASSES
from src.data.load_data import ebm_iter, pubmed_iter, load_doc_store, load_pico_store
from src.data.sampling import SamplerFactory


class ExtractionsDatamodule(L.LightningDataModule):
    """Datamodule for batch extraction and pre encoding of lucene index."""

    def __init__(
        self,
        cfg: SciDocDataConfig,
        tokenizer,
    ):
        super().__init__()
        self.batch_size = cfg.batch_size
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.categories = ["population", "intervention", "comparator", "outcome"]

    def setup(self, stage: str):
        # drop title and abstract to reduce data movement
        dataset = load_dataset(
            "parquet",
            data_files=str(DATAPATH / "index-pubmed" / "extractions.parquet"),
            split="train",
        ).remove_columns(["title", "abstract"])
        split = dataset.train_test_split(
            train_size=self.cfg.train_ratio, seed=self.cfg.seed
        )
        self.train_data, self.val_data = split["train"], split["test"]
        self.doc_store, self.doc_lookup = load_doc_store(self.cfg.text_embedder)

        self.pico_store, self.pico_lookup = {}, {}
        for c in self.categories:
            store, lookup = load_pico_store(c, self.cfg.text_embedder)
            self.pico_store[c] = store
            self.pico_lookup[c] = lookup

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
        doc_idx = []
        queries = []
        classes = []
        for row in rows:
            r_queries = []
            r_classes = []
            for c, c_int in QUERY_CLASSES.items():
                for q in row[c]:
                    # TODO maybe rewrite to batch lookup. unlikely to matter, <= 5 calls at a time
                    q_idx = self.pico_lookup[c][q]
                    q_embed = self.pico_store[c].reconstruct(q_idx)
                    r_queries.append(torch.from_numpy(q_embed))
                    r_classes.append(c_int)
            if len(r_queries) == 0:
                continue
            doc_idx.append(self.doc_lookup[row["pmid"]])
            queries.append(torch.stack(r_queries))
            classes.append(torch.tensor(r_classes))

        docs = self.doc_store.reconstruct_batch(doc_idx)
        queries = pad_sequence(queries, batch_first=True)
        classes = pad_sequence(classes, batch_first=True, padding_value=-1)
        # shapes: [B, D], [B, N, D], [B,N]
        return torch.from_numpy(docs), queries, classes


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
