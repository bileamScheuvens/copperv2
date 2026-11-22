from src.models.text_embedders import PromptRepsModel, SentenceTransformerModel
from src.utils.configs import TextEmbedType


def TextEncoderFactory(cfg):
    if cfg.text_embed_type == TextEmbedType.SENTENCE:
        return SentenceTransformerModel(cfg.text_embedder.value)
    if cfg.text_embed_type == TextEmbedType.PROMPTREPS:
        return PromptRepsModel(cfg.text_embedder.value)
    raise NotImplementedError(f"No text encoder of type {cfg.text_embed_type}")
