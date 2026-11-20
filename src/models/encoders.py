import lightning as L

from src.models.model_heads import PointEncoder, ProbabilisticEncoder
from src.utils.configs import EncoderHeadConfig


class EncoderHead(L.LightningModule):
    def __init__(
        self,
        cfg: EncoderHeadConfig,
        embed_dim: int,
        shared_dim: int,
        frozen_encoder,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.cfg = cfg
        self.shared_dim = shared_dim
        self.head = self.init_head(embed_dim)
        self.frozen_encoder = frozen_encoder

    def init_head(self, in_dim):
        if self.cfg.is_prob:
            # turn point embeddings to gaussian
            return ProbabilisticEncoder(in_dim, self.cfg.hidden_dim, self.shared_dim)
        else:
            return PointEncoder(
                in_dim,
                self.cfg.hidden_dim,
                self.shared_dim,
            )

    @property
    def embed_type(self):
        if self.cfg.is_prob:
            return "prob"
        return "point"

    def forward(self, x):
        """During training samples are pre encoded, only head applied.
        Cache misses / test time: also run encoder"""
        breakpoint()  # check dims
        return self.head(self.frozen_encoder(x))
