from src.utils.configs import TextEmbedder
import math
import os
from collections import defaultdict, Counter

from time import perf_counter
from collections import defaultdict

import faiss
import polars as pl
from tqdm import tqdm

from src.constants import INDEXPATH, QUERY_CLASSES, SHARDPATH
from src.data.load_data import load_extractions, load_negatives
from src.models.copper import CoPPeR

import numpy as np
import faiss
from time import perf_counter
from collections import defaultdict

search_mode = "batch"


def _shard_path(index_name, shard_i):
    os.makedirs(SHARDPATH, exist_ok=True)
    return SHARDPATH / f"{index_name}_shard_{shard_i}.faiss"


def _index_path(index_name):
    os.makedirs(INDEXPATH / "eval", exist_ok=True)
    return INDEXPATH / "eval" / f"{index_name}.faiss"


def eval_index_shard(model: CoPPeR, index_name, shard_i, k_shards):
    """Build shard <shard_i> from pubmed. Blockwise and atomic operation."""
    data_iter, n_documents = pubmed_iter()
    BLOCKSIZE = math.ceil(n_documents / k_shards)
    shard_start = shard_i * BLOCKSIZE

    index_shard = faiss.IndexFlatIP(model.cfg.paper_embedder.shared_dim)
    for i, (pmid, title, abstract) in tqdm(
        enumerate(data_iter), desc=f"Building shard {shard_i}.", total=n_documents
    ):
        # check if inside shard boundaries
        if not shard_start <= i < shard_start + BLOCKSIZE:
            continue
        index_shard.add(model.embed_paper(title, abstract).numpy())
    faiss.write_index(index_shard, str(_shard_path(index_name, shard_i)))


def eval_index_load(index_name, k_shards, clear=False):
    """Merge all shards into one index."""
    data_iter, n_documents = pubmed_iter()
    full_path = str(_index_path(index_name))

    # load if merged version exists
    if os.path.exists(full_path) and not clear:
        index = faiss.read_index(str(full_path))
    else:
        # otherwise load first shard and merge others
        index = faiss.read_index(str(_shard_path(index_name, 0)))

        for shard_i in range(1, k_shards):
            index_shard = faiss.read_index(str(_shard_path(index_name, shard_i)))
            index.merge_from(index_shard)
        # write to disk
        faiss.write_index(index, full_path)

    # metadata for reverse mapping
    idx_to_pmid = {}
    pmid_to_content = {}
    for i, (pmid, title, abstract) in enumerate(data_iter):
        idx_to_pmid[i] = pmid
        pmid_to_content[pmid] = (i, title, abstract)
    return index, idx_to_pmid, pmid_to_content


class PICOIndex:
    """
    Index over all extracted pico elements, embedded with a language model. Implements set similarity:
    sim(docA, docB) = sum(mean(sim(docA_P, docB_P)), sim(docA_I, docB_I),...)
    """

    def __init__(
        self,
        pico_store,
        pico_lookup,
        n_candidates=25,
        n_subcandidates=25,
    ):
        self.n_candidates, self.n_subcandidates = n_candidates, n_subcandidates

        self.raw_data = load_extractions()

        self.pmid2idx = {pmid: i for i, pmid in enumerate(self.raw_data["pmid"])}
        self.idx2pmid = {}
        self.pico_store = pico_store
        self.pico_lookup = pico_lookup
        for c in QUERY_CLASSES:
            # build forward and reverse indices
            idx2pmid = defaultdict(list)
            for pmid, elements in self.raw_data.select(["pmid", c]).iter_rows():
                for el in elements:
                    idx2pmid[pico_lookup[c][el]].append(pmid)

            self.idx2pmid[c] = idx2pmid

    def get_similar_doc(self, anchor_pmid):
        """Get similar doc by comparing picos."""

        candidates = defaultdict(float)

        # Loop over P I C O
        for pico_cat in QUERY_CLASSES:
            store = self.pico_store[pico_cat]
            lookup = self.pico_lookup[pico_cat]
            idx2pmid = self.idx2pmid[pico_cat]
            candidates_per_category = defaultdict(float)
            elements = self.raw_data[pico_cat][self.pmid2idx[anchor_pmid]].to_list()
            # For each extracted P, I, C, O
            for anchor_pico in elements:
                # search for n_subcandidates most similar ones and aggregate similarities
                # find idx of anchor element and reconstruct
                anchor_pico_vec = store.reconstruct(lookup[anchor_pico])
                # similarities to and idx of neighbor
                sims, nb_indices = store.search(
                    anchor_pico_vec[None, ...], self.n_subcandidates
                )
                for sim, nb_idx in zip(sims[0], nb_indices[0]):
                    # for candidate_pmid
                    for candidate in idx2pmid[nb_idx]:
                        # store dataset index with sim
                        candidates_per_category[self.pmid2idx[candidate]] += sim
            # transfer mean sim for each doc to global_candidates
            for c, sim in candidates_per_category.items():
                candidates[c] += sim / len(elements)
        # prevent retrieving itself
        candidates[self.pmid2idx[anchor_pmid]] = 0.0
        # return top candidates docs
        sorted_candidates = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
        return list(dict(sorted_candidates[: self.n_candidates]).keys())


# class CachedPICOIndex(PICOIndex):
#     def __init__(
#         self,
#         pico_store,
#         pico_lookup,
#         n_candidates=25,
#         n_subcandidates=25,
#     ):
#         self.pico_store = pico_store
#         self.pico_lookup = pico_lookup
#         self.n_candidates = n_candidates
#         self.n_subcandidates = n_subcandidates
#         self.index_path = INDEXPATH / "hard_negatives.json"
#         self.cache_all()
#
#     def cache_all(self):
#         if self.index_path.exists():
#             self.cached_candidates = load_negatives()
#             return
#
#         super().__init__(
#             self.pico_store, self.pico_lookup, self.n_candidates, self.n_subcandidates
#         )
#
#         negative_rows = []
#         for pmid in tqdm(self.raw_data["pmid"], desc="Caching hard negatives..."):
#             negative_rows.append(super().get_similar_doc(pmid))
#
#         self.cached_candidates = self.raw_data.select("pmid").with_columns(
#             pl.Series("negatives", negative_rows)
#         )
#         self.cached_candidates.write_parquet(self.index_path)
#
#     def get_similar_doc(self, anchor_pmid):
#         breakpoint()
#         pass


class SimplePICOIndex(PICOIndex):
    def __init__(
        self,
        pico_store: dict[str, faiss.Index],
        pico_lookup: dict[str, int],
        n_candidates=25,
        n_subcandidates=5,
        text_embedder: TextEmbedder = TextEmbedder.BGE,
    ):
        self.pico_store = pico_store
        self.pico_lookup = pico_lookup
        self.n_candidates = n_candidates
        self.n_subcandidates = n_subcandidates
        self.text_embedder = text_embedder

        self.extractions = load_extractions(sorted=True)
        # map element to pmids it occurs in
        self.el2pmid = {}
        for pico_cat in QUERY_CLASSES:
            self.el2pmid[pico_cat] = self.populate_el2pmid(pico_cat)
        self.negatives = {}
        # populates self.negatives as follows:
        # self.negatives[P]['adults'] -> [<idx of adolescents embedding in pico_store[P]>, ...]
        if not self.try_load():
            self.cache_all()

    def index_path(self, pico_cat):
        return (
            INDEXPATH
            / "negatives"
            / f"{pico_cat}_negatives_{self.text_embedder.name}.parquet"
        )

    def populate_el2pmid(self, pico_cat):
        # explode
        df = self.extractions.select("pmid", pico_cat).explode(pico_cat)
        # group
        df = df.group_by(pico_cat).agg(pl.col("pmid"))
        return {el: pmid for (el, pmid) in df.iter_rows()}

    def cache_single(self, pico_cat, batch_size=4096):
        store = self.pico_store[pico_cat]
        lookup = self.pico_lookup[pico_cat]
        exploded = self.extractions[pico_cat].explode(empty_as_null=False).unique()
        exploded_idx = exploded.replace_strict(lookup)

        rows = []
        for batch_start in tqdm(
            range(0, len(exploded), batch_size),
            desc=f"Populating {pico_cat} negatives...",
        ):
            batch = exploded_idx.slice(batch_start, batch_size).to_list()
            vecs = store.reconstruct_batch(batch)
            # self is always the most similar, search for one more and ignore first res
            _, nb_idx = store.search(vecs, self.n_subcandidates + 1)
            rows.append(nb_idx[:, 1:])

        negatives = exploded.to_frame().with_columns(
            pl.Series("nb_idx", np.concat(rows))
        )
        self.negatives[pico_cat] = negatives
        negatives.write_parquet(self.index_path(pico_cat))

    def try_load(self):
        for pico_cat in QUERY_CLASSES:
            index_path = self.index_path(pico_cat)
            if not index_path.exists():
                return False
            self.negatives[pico_cat] = pl.read_parquet(index_path).sort(pico_cat)
        return True

    def cache_all(self):
        for pico_cat in QUERY_CLASSES:
            self.cache_single(pico_cat)
        negative_rows = []
        for pmid in tqdm(self.raw_data["pmid"], desc="Caching hard negatives..."):
            negative_rows.append(super().get_similar_doc(pmid))

        self.cached_candidates = self.raw_data.select("pmid").with_columns(
            pl.Series("negatives", negative_rows)
        )
        self.cached_candidates.write_parquet(self.index_path)

    def get_similar_doc(self, anchor_pmid):
        candidates = Counter()
        anchor_row = self.extractions.row(
            self.extractions["pmid"].search_sorted(anchor_pmid), named=True
        )
        for pico_cat in QUERY_CLASSES:
            negatives = self.negatives[pico_cat]
            for el in anchor_row[pico_cat]:
                candidates.update(negatives[negatives[pico_cat].search_sorted(el), 1])

        return sorted(candidates, key=candidates.get, reverse=True)[: self.n_candidates]  # ty: ignore[no-matching-overload]
