from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM


@dataclass
class PLLScorer:
    """
    Pseudo-Log-Likelihood (PLL) scorer for masked language models.

    Sentence PLL = sum_i log p(token_i | sentence with token_i masked)
    This provides a simple, training-free proxy that often correlates with
    acceptability judgements, and is a natural starting point for model analysis.
    """
    model_name: str = "xlm-roberta-base"
    device: str | None = None

    def __post_init__(self) -> None:
        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(self.model_name)
        self.mdl = AutoModelForMaskedLM.from_pretrained(self.model_name).to(self.device)
        self.mdl.eval()
        self.special_ids = set(self.tok.all_special_ids)
        if self.tok.mask_token_id is None:
            raise ValueError("Model must be a masked LM with a mask token.")

    @torch.no_grad()
    def score(self, sentence: str) -> float:
        input_ids, attn = self._encode(sentence)
        ids = input_ids[0].tolist()
        pll = 0.0
        for i in range(1, input_ids.size(1) - 1):
            if ids[i] in self.special_ids:
                continue
            masked = input_ids.clone()
            masked[0, i] = self.tok.mask_token_id
            logits = self.mdl(masked, attention_mask=attn).logits
            log_probs = torch.log_softmax(logits[0, i], dim=-1)
            pll += float(log_probs[ids[i]].item())
        return pll

    @torch.no_grad()
    def token_logprobs(self, sentence: str) -> List[Tuple[str, float]]:
        """
        Returns per-token log-prob contributions used in PLL.
        Useful for simple interpretability and qualitative error analysis.
        """
        input_ids, attn = self._encode(sentence)
        ids = input_ids[0].tolist()
        toks = self.tok.convert_ids_to_tokens(ids)

        contribs: List[Tuple[str, float]] = []
        for i in range(1, input_ids.size(1) - 1):
            if ids[i] in self.special_ids:
                continue
            masked = input_ids.clone()
            masked[0, i] = self.tok.mask_token_id
            logits = self.mdl(masked, attention_mask=attn).logits
            log_probs = torch.log_softmax(logits[0, i], dim=-1)
            contribs.append((toks[i], float(log_probs[ids[i]].item())))
        return contribs

    def _encode(self, sentence: str):
        enc = self.tok(sentence, return_tensors="pt")
        return enc["input_ids"].to(self.device), enc["attention_mask"].to(self.device)
