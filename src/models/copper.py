import lightning as L
import torch
from transformers import AutoTokenizer

from src.losses import get_fitting_criterion
from src.metrics import mean_l2, mean_sim
from src.metrics.plots import plot_means
from src.models.combiners import Combiner
from src.models.encoders import EncoderHead
from src.models.text_embedders import SentenceTransformerModel
from src.utils.configs import CoPPeRConfig


class CoPPeR(L.LightningModule):
    """COmpositional Probabilistic Pretraining for systEmatic Reviews"""

    def __init__(self, cfg=CoPPeRConfig, load_encoder=True):
        super().__init__()
        self.cfg = cfg
        self.strict_loading = False

        if load_encoder:
            self.text_encoder = SentenceTransformerModel(cfg.text_encoder.model_id)
        else:
            self.text_encoder = lambda x: x

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.text_encoder.model_id.value)

        # query encoder
        self.query_encoder = EncoderHead(
            cfg.query_encoder,
            embed_dim=cfg.text_encoder.embed_dim,
            shared_dim=cfg.shared_dim,
            frozen_encoder=self.text_encoder,
        )
        self.pico_combiner = Combiner(
            cfg.combiner,
            shared_dim=cfg.shared_dim,
            use_prob_encoder=cfg.query_encoder.is_prob,
        )

        # doc encoder
        self.doc_encoder = EncoderHead(
            cfg.doc_encoder,
            embed_dim=cfg.text_encoder.embed_dim,
            shared_dim=cfg.shared_dim,
            frozen_encoder=self.text_encoder,
        )

        # get appropriate criterion depending on if embeddings are probabilistic
        self.retrieval_loss = get_fitting_criterion(
            query_prob=cfg.query_encoder.is_prob,
            doc_prob=cfg.doc_encoder.is_prob,
            temperature=1,
        )
        self.save_hyperparameters()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.cfg.lr)

    def join_text(self, title, abstract):
        return f"{title}\n\n{abstract}"

    @torch.no_grad
    def embed_query(self, query):
        pico_individual, labels = self.pico_extractor.encode_pico(pico)
        pico_projected = self.pico_projector(pico_individual)
        means = self.pico_combiner(pico_projected, labels)["mean"]

        breakpoint()  # double check dims
        return means.unsqueeze(0).cpu()

    @torch.no_grad()
    def embed_paper(self, title, abstract):
        doc_embed = self.doc_encoder(self.join_text(title, abstract))
        return doc_embed.cpu()  # TODO should this be cpu?

    def forward(self, docs, queries, query_classes):
        """Shapes: (B, shared_dim) | (B, max_elements, shared_dim) | (B, max_elements, 1)"""

        preds = {}
        if self.cfg.doc_encoder.is_prob:
            docs = self.doc_encoder(docs)  # [B, 2, shared_dim]
            preds["doc_means"] = docs[..., 0, :]
            preds["doc_logvars"] = docs[..., 1, :]

        else:
            preds["doc_means"] = self.doc_encoder(docs)  # [B, shared_dim]

        queries = self.query_encoder(queries)  # [B, (2), shared_dim]
        queries_composed = self.pico_combiner(
            queries, query_classes
        )  # TODO handle batching in compositor

        preds["query_means"] = queries_composed["mean"]
        if self.cfg.query_encoder.is_prob:
            preds["query_logvars"] = queries_composed["logvar"]
            preds["query_zs"] = queries_composed["log_z"]
        return preds

    def _compute_loss(
        self,
        doc_means,
        query_means,
        doc_logvars=None,
        query_logvars=None,
        query_zs=None,
        prefix="train",
    ):
        # TODO bidirectional retrieval loss
        retrieval_loss, recall = self.retrieval_loss.forward(
            query=query_means,
            query_logsigma=query_logvars,
            query_z=query_zs,
            target=doc_means,
            target_logsigma=doc_logvars,
            target_z=None,
            recall=True,
        )
        loss = retrieval_loss
        metrics = {
            f"{prefix}/retrieval_loss": retrieval_loss,
            f"{prefix}/recall": recall,
        }
        if query_logvars is not None:
            l2_pico = self.cfg.l2_lambda * ((query_logvars) ** 2).mean()
            metrics[f"{prefix}/l2_pico"] = l2_pico
            metrics[f"{prefix}/mean_pico_var"] = query_logvars.exp().mean()
            loss += l2_pico
        if doc_logvars is not None:
            l2_paper = self.cfg.l2_lambda * ((doc_logvars) ** 2).mean()
            metrics[f"{prefix}/l2_paper"] = l2_paper
            metrics[f"{prefix}/mean_paper_var"] = doc_logvars.exp().mean()
            loss += l2_paper
        metrics[f"{prefix}/loss"] = loss
        return metrics

    def training_step(self, batch, batch_idx):  # TODO manage batch routing
        metrics = self._compute_loss(**self(*batch))
        self.log_dict(metrics, batch_size=len(batch[0]))
        return metrics["train/loss"]

    def on_validation_start(self):
        self.umap_buffer = {"query_means": [], "doc_means": [], "titles": []}
        self.val_batch_count = 0

    def validation_step(self, batch, batch_idx):
        self.val_batch_count += 1
        preds = self(*batch)
        metrics = self._compute_loss(
            **preds,
            prefix="val",
        )
        mean_l2_correct, mean_l2_incorrect = mean_l2(
            preds["doc_means"], preds["query_means"]
        )
        mean_sim_correct, mean_sim_incorrect = mean_sim(
            preds["doc_means"], preds["query_means"]
        )
        metrics["val/mean_l2_correct"] = mean_l2_correct
        metrics["val/mean_l2_incorrect"] = mean_l2_incorrect
        metrics["val/mean_sim_correct"] = mean_sim_correct
        metrics["val/mean_sim_incorrect"] = mean_sim_incorrect
        self.log_dict(metrics, batch_size=len(batch[0]))
        if self.val_batch_count < 3:
            self.umap_buffer["query_means"].append(preds["query_means"])
            self.umap_buffer["doc_means"].append(preds["doc_means"])
            self.umap_buffer["titles"].extend(batch[0])
        return metrics["val/loss"]

    def on_validation_end(self):
        fig = plot_means(
            torch.cat(self.umap_buffer["query_means"]),
            torch.cat(self.umap_buffer["doc_means"]),
            paper_titles=self.umap_buffer["titles"],
        )
        # fig_img = Image.open(io.BytesIO(fig.to_image(format="png")))
        self.logger.experiment.log({"mean_umap": fig})  # ty:ignore[unresolved-attribute]
