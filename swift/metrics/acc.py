# Copyright (c) ModelScope Contributors. All rights reserved.
import numpy as np
import torch
from transformers.trainer_utils import EvalPrediction
from typing import Dict, List, Literal

from swift.utils import Serializer
from .base import EvalMetrics


def _normalize_generate_text(text: str) -> str:
    text = text.strip()
    if '</think>' in text:
        text = text.split('</think>', 1)[-1]
    return text.strip()


def compute_acc(preds,
                labels,
                *,
                acc_strategy: Literal['token', 'seq'] = 'token',
                is_encoder_decoder: bool = False,
                cu_seqlens=None) -> Dict[str, List[float]]:

    if isinstance(preds, torch.Tensor):
        if torch.is_floating_point(labels):
            return {}
        preds = preds.cpu().numpy()
        labels = labels.cpu().numpy()
    if preds.ndim >= 2 and not is_encoder_decoder:
        labels = labels[..., 1:]
        preds = preds[..., :-1]
    if np.issubdtype(labels.dtype, np.floating) or preds.shape != labels.shape:
        return {}

    masks = labels != -100
    if acc_strategy == 'token' or preds.ndim == 1:  # 'single_label_classification'
        acc_list = (preds[masks] == labels[masks]).tolist()
    else:
        acc_list = []
        if cu_seqlens is not None and masks.shape[0] == 1:
            # padding_free
            for i in range(cu_seqlens.shape[0] - 1):
                start, end = cu_seqlens[i], cu_seqlens[i + 1]
                acc_list.append(np.all(preds[0, start:end] == labels[0, start:end]))
        else:
            for i, m in enumerate(masks):
                acc_list.append(np.all(preds[i, m] == labels[i, m]))
    return {f'{acc_strategy}_acc' if preds.ndim >= 2 else 'acc': acc_list}


class AccMetrics(EvalMetrics):

    def compute_metrics(self, eval_prediction: EvalPrediction) -> Dict[str, float]:
        if getattr(self.args, 'predict_with_generate', False):
            return self._compute_generate_acc(eval_prediction)
        metric = compute_acc(
            eval_prediction.predictions,
            eval_prediction.label_ids,
            acc_strategy=self.args.acc_strategy,
            is_encoder_decoder=self.trainer.is_encoder_decoder)
        if len(metric) == 0:
            return {}
        return {k: sum(v) / len(v) for k, v in metric.items()}

    def _compute_generate_acc(self, eval_prediction: EvalPrediction) -> Dict[str, float]:
        preds, labels = eval_prediction.predictions, eval_prediction.label_ids
        acc_strategy = self.args.acc_strategy
        acc_list = []
        for i in range(preds.shape[0]):
            pred = _normalize_generate_text(Serializer.from_tensor(preds[i]))
            label = _normalize_generate_text(Serializer.from_tensor(labels[i]))
            if acc_strategy == 'seq':
                acc_list.append(pred == label)
            else:
                acc_list.extend([p == l for p, l in zip(pred, label)])
        if not acc_list:
            return {}
        key = f'{acc_strategy}_acc' if acc_strategy == 'seq' else 'acc'
        return {key: sum(acc_list) / len(acc_list)}

    def preprocess_logits_for_metrics(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if getattr(self.args, 'predict_with_generate', False):
            return logits
        if isinstance(logits, (list, tuple)):
            logits = logits[0]
        preds = logits.argmax(dim=-1)
        return preds
