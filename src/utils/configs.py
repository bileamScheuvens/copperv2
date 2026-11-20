from dataclasses import dataclass
from enum import Enum

from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

cs = ConfigStore()


class SamplerType(Enum):
    RANDOM = "random_sampler"
    HARDNEGATIVE = "hard_negative_sampler"


@dataclass
class SamplerConfig:
    sampler_type: SamplerType


@dataclass
class HardNegativeConfig(SamplerConfig):
    sampler_type: SamplerType
    corpus: str
    extractor_model: str
    n_candidates: int
    n_subcandidates: int
    n_anchors: int
    n_per_anchor: int


cs.store(group="data/sampler", name="base_sampler", node=SamplerConfig)
cs.store(group="data/sampler", name="base_hardnegative", node=HardNegativeConfig)


@dataclass
class SciDocDataConfig:
    batch_size: int
    train_ratio: float
    seed: int
    sampler: SamplerConfig


cs.store(group="data", name="base_data", node=SciDocDataConfig)


@dataclass
class TrainerConfig:
    precision: str | None = None
    val_check_interval: int | None = None
    log_every_n_steps: int | None = None
    limit_train_batches: int | None = None
    limit_val_batches: int | None = None
    overfit_batches: int = 0
    max_time: str | None = None


cs.store(group="trainer", name="base_trainer", node=TrainerConfig)


@dataclass
class LoggerConfig:
    project: str
    mode: str | None = None
    space_id: str | None = None


cs.store(group="logger", name="base_logger", node=LoggerConfig)


class TextEmbedType(Enum):
    PROMPTREPS = "promptreps"
    SENTENCE = "sentence_transformer"


class TextEmbedder(Enum):
    PUBMEDBERT = "neuml/pubmedbert-base-embeddings"
    QWEN2 = "Qwen/Qwen2.5-1.5B-Instruct"
    QWEN3 = "Qwen/Qwen3.5-0.8B"
    QWEN3E = "Qwen/Qwen3-Embedding-0.6B"
    BGE = "BAAI/bge-large-zh-v1.5"
    E5 = "intfloat/multilingual-e5-large"


@dataclass
class PubMedPicoConfig:
    base_url: str
    hidden_dim: int
    max_len: int
    shared_dim: int
    text_embed_type: TextEmbedType
    text_embedder: TextEmbedder
    use_prob_encoder: bool
    considered_elements: list[str]
    cache_extraction: bool = False
    pico_dropout: float = 0.0


cs.store(group="model/pico_extractor", name="base_pubmed_pico", node=PubMedPicoConfig)


class PaperEmbedderClass(Enum):
    SPECTER = "specter"
    SPECTER2 = "specter2"
    PLAIN = "plain"


@dataclass
class TextEncoderConfig:
    base_url: str
    embed_dim: int


cs.store(group="model/text_encoder", name="base_text_encoder", node=TextEncoderConfig)


@dataclass
class EncoderHeadConfig:
    hidden_dim: int
    is_prob: bool


cs.store(group="model/doc_encoder", name="base_doc_encoder", node=EncoderHeadConfig)
cs.store(group="model/query_encoder", name="base_query_encoder", node=EncoderHeadConfig)


@dataclass
class PaperEmbedderConfig:
    model_class: PaperEmbedderClass
    base_url: str | None
    text_embed_type: TextEmbedType | None
    text_embedder: TextEmbedder | None
    hidden_dim: int
    shared_dim: int
    use_prob_encoder: bool
    adapter: str | None
    max_len: int = 512
    unfreeze_after: int = -1  # frozen by default


cs.store(
    group="model/paper_embedder", name="base_paper_embedder", node=PaperEmbedderConfig
)


class PicoAggType(Enum):
    SUM = "sum"
    ATTN = "attention"
    GAUSSIAN = "gaussian"
    HADAMARD = "hadamard"
    MLP = "mlp"


@dataclass
class PicoCombinerConfig:
    intra_agg: PicoAggType
    inter_agg: PicoAggType
    n_heads: int | None = 8
    use_prob_encoder: bool = False


cs.store(group="model/combiner", name="base_combiner", node=PicoCombinerConfig)


@dataclass
class CoPPeRConfig:
    temperature: int
    text_encoder: TextEncoderConfig
    query_encoder: EncoderHeadConfig
    doc_encoder: EncoderHeadConfig
    shared_dim: int
    combiner: PicoCombinerConfig
    lr: float
    l2_lambda: float
    pico_dropout: float = 0.2
    ckpt_path: str | None = None


cs.store(group="model", name="base_model", node=CoPPeRConfig)


class EvalMethods(Enum):
    EBM_NLP = "ebm_nlp"
    VIS = "visualisation"
    DASH = "dash"
    PROBE = "probing"
    TRANSFER = "transfer"
    SYSREV = "sysrev"
    SYSREV_SEED = "sysrev_seed"
    TIME = "time_demo"


@dataclass
class EvalConfig:
    model: CoPPeRConfig
    data: SciDocDataConfig
    experiment: dict | None
    eval_method: EvalMethods
    index_name: str = "index"
    N_vis: int = 128
    k_shards: int = 16
    shard_i: int | None = None


cs.store(name="base_eval", node=EvalConfig)


@dataclass
class ExperimentConfig:
    experiment_name: str
    model: CoPPeRConfig
    data: SciDocDataConfig
    trainer: TrainerConfig
    logger: LoggerConfig


cs.store(name="experiment_config", node=ExperimentConfig)


def as_dict(conf):
    return OmegaConf.to_container(conf, resolve=True)


def get_wandb_names(cfg):
    hc = HydraConfig.get()
    if hc.sweeper.params is None:
        return cfg.experiment_name, None
    swept_params = hc.sweeper.params.keys()
    group_name = "sweep_" + "_".join([x.split(".")[-1] for x in swept_params])
    experiment_name = f"{group_name}-{'_'.join([str(OmegaConf.select(cfg, x)) for x in swept_params])}"
    return experiment_name, group_name
