import argparse

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=lambda x: TextEmbedder[x],
        choices=list(TextEmbedder),
        default=TextEmbedder.BGE,
    )
    args = parser.parse_args()
    encode_abstracts(args.model)
