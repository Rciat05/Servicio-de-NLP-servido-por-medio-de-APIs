"""Loads the fine-tuned model and answers questions on CPU (or GPU for evaluation)."""
from __future__ import annotations

import json
import os
from pathlib import Path

from qa.features import encode_windows
from qa.postprocess import Analysis, WindowOutput, analyze_windows, decide


class QAModel:
    def __init__(self, model_dir, device: str = "cpu", threads: int | None = None):
        import torch
        from transformers import AutoModelForQuestionAnswering, AutoTokenizer

        self._torch = torch
        model_dir = Path(model_dir)
        cfg_path = model_dir / "qa_config.json"
        if not cfg_path.exists():
            raise FileNotFoundError(f"{cfg_path} not found. Did training finish? Is MODEL_DIR right?")
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        self.base_model: str = cfg["base_model"]
        self.max_seq_length: int = int(cfg.get("max_seq_length", 384))
        self.doc_stride: int = int(cfg.get("doc_stride", 128))
        self.threshold: float = float(cfg.get("threshold", 0.0))

        self.device = device
        if device == "cpu":
            n = threads or int(os.environ.get("QA_THREADS", 0)) or max(1, (os.cpu_count() or 2) // 2)
            torch.set_num_threads(n)

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
        if not self.tokenizer.is_fast:
            raise RuntimeError("A fast tokenizer is required (offset mapping).")
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_dir).to(device).eval()
        self.params: int = sum(p.numel() for p in self.model.parameters())

    def analyze(self, context: str, question: str) -> Analysis:
        """Best span + null score, BEFORE applying the threshold (used to tune the threshold)."""
        torch = self._torch
        inputs, offsets = encode_windows(
            self.tokenizer, question, context, self.max_seq_length, self.doc_stride
        )
        tensors = {k: torch.tensor(v, device=self.device) for k, v in inputs.items()}
        with torch.inference_mode():
            out = self.model(**tensors)
        start = out.start_logits.float().cpu().numpy()
        end = out.end_logits.float().cpu().numpy()
        windows = [WindowOutput(start[i], end[i], offsets[i]) for i in range(start.shape[0])]
        return analyze_windows(context, windows)

    def predict(self, context: str, question: str):
        """Returns (answer | None, start | None, score)."""
        return decide(self.analyze(context, question), self.threshold)

    def warmup(self) -> None:
        """First inference is slow (lazy init). Do it before the harness arrives."""
        self.predict("The Lempa River is the longest river in El Salvador.", "What is the longest river?")
