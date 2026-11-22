import argparse
import pickle

import faiss
import torch
from sentence_transformers import SentenceTransformer

from src.constants import INDEXPATH
from src.data.load_data import load_extractions
from src.utils.configs import TextEmbedder


@torch.no_grad()
def encode_abstracts(model_id: TextEmbedder, batch_size=32):
    df = load_extractions()
    model = SentenceTransformer(model_id.value)
    abstracts = [
        f"{title}\n\n{abstract}"
        for title, abstract in df.select("title", "abstract").iter_rows()
    ]
    embeds = model.encode(
        abstracts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    index = faiss.IndexFlatIP(embeds.shape[1])
    index.add(embeds)
    faiss.write_index(index, str(INDEXPATH / f"abstr_{model_id.name}.faiss"))


@torch.no_grad()
def encode_pico(model_id: TextEmbedder, target, batch_size=32):
    """Stores embeddings for extracted pico elements. Keeps str elements as parquet to build lookup from."""
    df = load_extractions()
    model = SentenceTransformer(model_id.value)
    exploded = df[target].explode(empty_as_null=False).unique()
    # TODO prefix category to element
    embeds = model.encode(
        exploded.to_list(),
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    index = faiss.IndexFlatIP(embeds.shape[1])
    index.add(embeds)
    faiss.write_index(index, str(INDEXPATH / f"{target}_{model_id.name}.faiss"))
    exploded.to_frame().write_parquet(INDEXPATH / f"{target}_str.parquet")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target",
        choices=["abstracts", "population", "intervention", "comparator", "outcome"],
        default="abstracts",
    )
    parser.add_argument(
        "--model",
        type=lambda x: TextEmbedder[x],
        choices=list(TextEmbedder),
        default=TextEmbedder.BGE,
    )
    args = parser.parse_args()
    if args.target == "abstracts":
        encode_abstracts(args.model)
    else:
        encode_pico(args.model, target=args.target)
