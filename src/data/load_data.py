import json
import os
from pathlib import Path

import faiss
import polars as pl

from src.constants import DATAPATH, INDEXPATH
from src.utils.configs import TextEmbedder


def load_extractions():
    df = pl.read_parquet(DATAPATH / "index-pubmed" / "extractions.parquet")
    return df


def load_pubmed():
    with open(DATAPATH / "pubmed" / "pubmed.json", "r") as f:
        return json.load(f)


def load_doc_store(model_id: TextEmbedder):
    stub = f"abstr_{model_id.name}"
    df = load_extractions()
    index = faiss.read_index(str(INDEXPATH / f"{stub}.faiss"))
    reverse_lookup = {x: i for i, x in enumerate(df["pmid"])}
    return index, reverse_lookup


def load_pico_store(category, model_id: TextEmbedder):
    # TODO write test for this
    df = pl.read_parquet(INDEXPATH / f"{category}_str.parquet")
    index = faiss.read_index(str(INDEXPATH / f"{category}_{model_id.name}.faiss"))
    reverse_lookup = {x: i for i, x in enumerate(df[category])}
    return index, reverse_lookup


def pubmed_iter():
    data = load_pubmed()

    def _pubmed_iter():
        for pmid, doc in data.items():
            if not doc["abstract"]:
                continue
            yield pmid, doc["title"], doc["abstract"]

    return _pubmed_iter(), len(data)


def load_ebm():
    docs = {}
    for file in os.scandir(DATAPATH / "ebm-nlp" / "ebm_nlp_2_00" / "documents"):
        file = Path(file)
        if file.suffix != ".txt":
            continue
        with open(file) as f:
            docs[file.stem] = f.read()
    return docs


def ebm_iter():
    data = load_ebm()

    def _ebm_iter():
        for pmid, doc in data.items():
            title, abstract = doc.split("\n\n", 1)
            yield pmid, title, abstract

    return _ebm_iter(), len(data)
