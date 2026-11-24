import faiss
import polars as pl

from src.constants import DATAPATH, INDEXPATH
from src.utils.configs import TextEmbedder


def load_negatives():
    df = pl.read_parquet(INDEXPATH / "negatives.parquet")
    return df


def load_extractions():
    df = pl.read_parquet(DATAPATH / "index-pubmed" / "extractions.parquet")
    return df


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
