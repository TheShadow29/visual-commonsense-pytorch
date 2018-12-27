import _init_paths
import json
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from dataclasses import dataclass
from pytorch_pretrained_bert.modeling import PreTrainedBertModel, BertConfig, BertModel, BertPooler
from pytorch_pretrained_bert.modeling import BertEmbeddings, BertEncoder, BertForMultipleChoice
from typing import List, Dict, Union, Any, Optional
from dat_loader import get_bert_data, InputFeatures


class VCRBert(PreTrainedBertModel):
    def __init__(self, bert_cfg: BertConfig, cfg: Optional[Dict[str, Any]] = None):
        self.bert_cfg = bert_cfg
        super().__init__(bert_cfg)
        self.cfg = cfg
        self.embeddings = BertEmbeddings(bert_cfg)
        self.encoder = BertEncoder(bert_cfg)
        self.pooler = BertPooler(bert_cfg)
        self.num_choices = 4
        self.classifier = nn.Linear(bert_cfg.hidden_size, 1)
        self.apply(self.init_bert_weights)

    def forward(self, inp: Dict[str, torch.tensor]) -> torch.tensor:
        input_ids = inp['input_ids']
        segment_ids = inp['segment_ids']
        input_mask = inp['input_mask'].float()
        flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        flat_segment_ids = segment_ids.view(-1, segment_ids.size(-1))
        flat_input_mask = input_mask.view(-1, input_mask.size(-1))
        extended_attention_mask = flat_input_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        embedding_output = self.embeddings(flat_input_ids, flat_segment_ids)
        encoded_layers = self.encoder(embedding_output,
                                      extended_attention_mask,
                                      output_all_encoded_layers=True)
        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output)

        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, self.num_choices)
        return reshaped_logits


class LossB(nn.Module):
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = cfg
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, out: torch.tensor, inp: Dict[str, Any]) -> torch.tensor:
        targ = inp['target_labels']
        return self.loss_fn(out, targ)


class EvalB(nn.Module):
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = cfg

    def forward(self, out: torch.tensor, inp: Dict[str, Any]) -> torch.tensor:
        targ = inp['target_labels']
        pred_logits = F.log_softmax(out, dim=-1)
        pred_labels = torch.argmax(pred_logits, dim=-1)
        return (targ == pred_labels).float().mean()


if __name__ == '__main__':
    cfg = json.load(open('./cfg.json'))
    data = get_bert_data(cfg)
    mdl = VCRBert.from_pretrained(cfg['bert_model'], cfg=cfg)
    batch = next(iter(data.train_dl))
    out = mdl(batch)
