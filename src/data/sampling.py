from __future__ import annotations

from random import randrange, seed

from datasets.arrow_dataset import Dataset
from torch.utils.data import BatchSampler, RandomSampler, Sampler

from src.data.indexing import SimplePICOIndex
from src.utils.configs import HardNegativeConfig, SamplerType


def SamplerFactory(cfg, dataset, pico_store=None, pico_lookup=None):
    seed(cfg.seed)

    if cfg.sampler.sampler_type == SamplerType.RANDOM:
        return BatchSampler(
            RandomSampler(dataset), batch_size=cfg.batch_size, drop_last=False
        )
    if cfg.sampler.sampler_type == SamplerType.HARDNEGATIVE:
        return HardNegativeSampler(
            cfg.sampler, pico_store, pico_lookup, cfg.batch_size, dataset
        )

    raise NotImplementedError(f"No sampler of type {cfg.sampler.sampler_type}")


class HardNegativeSampler(Sampler):
    def __init__(
        self,
        cfg: HardNegativeConfig,
        pico_store,
        pico_lookup,
        batch_size: int,
        dataset: Dataset,
    ):
        self.cfg = cfg
        self.n_anchors = cfg.n_anchors
        self.n_per_anchor = cfg.n_per_anchor
        self.batch_size = batch_size
        self.index = SimplePICOIndex(
            pico_store=pico_store,
            pico_lookup=pico_lookup,
            n_candidates=cfg.n_candidates,
            n_subcandidates=cfg.n_subcandidates,
        )
        self.dataset = dataset
        self.pmid2idx = {pmid: i for i, pmid in enumerate(dataset["pmid"])}

    def __len__(self):
        return len(self.dataset) // self.batch_size

    def sample_idx(self):
        return randrange(len(self.dataset))

    def __iter__(self):
        """
        One Batch consists of n_a anchors, n_n negatives per anchor and batch_size-(n_a * (n_n+1)) random samples.
        Returns list of indices
        """
        for _ in range(len(self)):
            batch_indices = set()
            # gather anchor and hard negatives
            for _ in range(self.n_anchors):
                # pick anchor from subset
                anchor_idx = self.sample_idx()
                batch_indices.add(anchor_idx)
                # get negative candidates, skipping those from val test
                negatives = [
                    self.pmid2idx[x]
                    for x in self.index.get_similar_doc(
                        self.dataset[anchor_idx]["pmid"]
                    )
                    if x in self.pmid2idx
                ]
                # add to batch
                batch_indices |= set(negatives[: self.n_per_anchor])
            # fill batch with random negatives ensuring no duplicates
            while len(batch_indices) < self.batch_size:
                batch_indices.add(self.sample_idx())
            yield batch_indices
