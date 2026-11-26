import lightning as L
import torch
from datasets import load_dataset, Value
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from src.constants import DATAPATH, QUERY_CLASSES
from src.data.load_data import load_doc_store, load_pico_store
from src.data.sampling import SamplerFactory
from src.utils.configs import PubMedDataConfig


class PubMedDatamodule(L.LightningDataModule):
    """Datamodule for batch extraction and pre encoding of lucene index."""

    def __init__(
        self,
        cfg: PubMedDataConfig,
    ):
        super().__init__()
        self.batch_size = cfg.batch_size
        self.cfg = cfg

    def setup(self, stage: str):
        # drop title and abstract to reduce data movement
        dataset = (
            load_dataset(
                "parquet",
                data_files=str(DATAPATH / "index-pubmed" / "extractions.parquet"),
                split="train",
            )
            .remove_columns(["title", "abstract"])
            .cast_column("pmid", Value("int64"))
        )
        split = dataset.train_test_split(
            train_size=self.cfg.train_ratio, seed=self.cfg.seed
        )
        self.train_data, self.val_data = split["train"], split["test"]
        self.doc_store, self.doc_lookup = load_doc_store(self.cfg.text_embedder)

        self.pico_store, self.pico_lookup = {}, {}
        for c in QUERY_CLASSES:
            store, lookup = load_pico_store(c, self.cfg.text_embedder)
            self.pico_store[c] = store
            self.pico_lookup[c] = lookup

    def train_dataloader(self):
        sampler = SamplerFactory(
            self.cfg,
            self.train_data,
            pico_store=self.pico_store,
            pico_lookup=self.pico_lookup,
        )
        return DataLoader(
            self.train_data,
            batch_sampler=sampler,
            collate_fn=self.collate,
            num_workers=self.cfg.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            collate_fn=self.collate,
            num_workers=self.cfg.num_workers,
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
