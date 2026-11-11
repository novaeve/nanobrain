"""
Training engine for HEIST cell-type annotation.

Extends ScGPTTrainerEngine to pass spatial coordinates and batch_id
to the model's forward method, which needs them for on-the-fly graph
construction.
"""
import torch
import torch.nn.functional as F
from src.engines.scgpt_trainer import ScGPTTrainerEngine


class HEISTTrainerEngine(ScGPTTrainerEngine):

    def compute_loss(self, batch_data):
        X          = batch_data["X"].to(self.device)
        spatial    = batch_data["spatial"].to(self.device)
        batch_id   = batch_data["batch_id"].to(self.device)
        y_c        = batch_data["y_class"].to(self.device)
        y_sc       = batch_data["y_subclass"].to(self.device)
        y_st       = batch_data["y_supertype"].to(self.device)
        confidence = batch_data["confidence"].to(self.device)

        logits, cell_emb = self.model(X, spatial=spatial, batch_id=batch_id)
        logit_c, logit_sc, logit_st = logits

        loss_c  = F.cross_entropy(logit_c, y_c)
        loss_sc = F.cross_entropy(logit_sc, y_sc)
        loss_st = (F.cross_entropy(logit_st, y_st, reduction="none") * confidence).mean()

        total_loss = loss_c + loss_sc + loss_st

        metrics = {
            "total":     total_loss.item(),
            "class":     loss_c.item(),
            "subclass":  loss_sc.item(),
            "supertype": loss_st.item(),
        }
        logits_dict = {
            "class":     logit_c,
            "subclass":  logit_sc,
            "supertype": logit_st,
        }
        return total_loss, metrics, logits_dict, cell_emb
