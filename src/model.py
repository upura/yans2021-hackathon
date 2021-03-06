from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

from dataset.ner import NerDataset


def create_pooler_matrix(
    input_ids: torch.Tensor, word_idxs: List[List[int]], pool_type="head"
):
    bsz, subword_len = input_ids.size()
    max_word_len = max([len(w) for w in word_idxs])
    pooler_matrix = torch.zeros(bsz * max_word_len * subword_len)

    if pool_type == "head":
        pooler_idxs = [
            subword_len * max_word_len * batch_offset + subword_len * word_offset + w
            for batch_offset, word_idx in enumerate(word_idxs)
            for word_offset, w in enumerate(word_idx[:-1])
        ]
        pooler_matrix.scatter_(0, torch.LongTensor(pooler_idxs), 1)
        return pooler_matrix.view(bsz, max_word_len, subword_len)

    elif pool_type == "average":
        pooler_idxs = [
            subword_len * max_word_len * batch_offset
            + subword_len * word_offset
            + w / (word_idx[word_offset + 1] - word_idx[word_offset])
            for batch_offset, word_idx in enumerate(word_idxs)
            for word_offset, _ in enumerate(word_idx[:-1])
            for w in range(word_idx[word_offset], word_idx[word_offset + 1])
        ]
        pooler_matrix.scatter_(0, torch.LongTensor(pooler_idxs), 1)
        return pooler_matrix.view(bsz, max_word_len, subword_len)


class BertForMultilabelNER(nn.Module):
    def __init__(
        self,
        bert: BertModel,
        attribute_num: int,
        dropout: float = 0.1
    ):
        super().__init__()
        self.bert: BertModel = bert
        self.dropout = nn.Dropout(dropout)
        self.num_attributes: int = attribute_num

        # classifier that classifies token into IOB tag (B, I, O) for each attribute
        # self.output_layer = nn.Linear(768, 768 * attribute_num)

        self.relu = nn.ReLU()

        # classifier that classifies token into IOB tag (B, I, O) for each attribute
        bert_hidden_size = self.bert.config.hidden_size
        self.classifier = nn.Linear(bert_hidden_size, self.num_attributes * 3)

    def forward(
        self,
        input_ids: torch.Tensor,  # (b, seq)
        labels,  # (b, word, attr) or None
        confidences,  # (b, word, attr) or None
        pooling_matrix,  # (b, word, seq)
        **_,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        batch_size, word_len, sequence_len = pooling_matrix.size()

        bert_out = self.bert(input_ids, attention_mask=(input_ids > 0))
        # create word-level representations using pooler matrix
        # (b, word, seq), (b, seq, hid) -> (b, word, hid)
        sequence_output = torch.bmm(pooling_matrix, bert_out.last_hidden_state)
        sequence_output = self.dropout(sequence_output)  # (b, word, hid)

        # hiddens = [self.relu(layer(sequence_output)) for layer in self.output_layer]
        # (b, word, hid) -> (b, word, attr*3) -> (b, word, attr, 3)
        logits = self.classifier(sequence_output).view(batch_size, word_len, self.num_attributes, 3)

        loss = None
        if labels is not None:
            raw_loss = F.cross_entropy(
                logits[:, :-1].permute(0, 3, 1, 2),
                labels,
                reduction="none",
                ignore_index=NerDataset.PAD_FOR_LABELS,
            )  # (b, word, attr)
            loss = (raw_loss * confidences).sum() / (confidences.sum() + 1e-6)

        return loss, logits


class BertForMultilabelNER2(nn.Module):
    def __init__(
        self,
        bert: BertModel,
        attribute_num: int,
        dropout: float = 0.1
    ):
        super().__init__()
        self.bert: BertModel = bert
        self.dropout = nn.Dropout(dropout)
        self.num_attributes: int = attribute_num

        # classifier that classifies token into IOB tag (B, I, O) for each attribute
        bert_hidden_size = self.bert.config.hidden_size
        self.output_layer = nn.Linear(bert_hidden_size, self.num_attributes * bert_hidden_size)

        self.relu = nn.ReLU()

        # classifier that classifies token into IOB tag (B, I, O) for each attribute
        self.classifier = nn.Linear(bert_hidden_size, 3)

    def forward(
        self,
        input_ids: torch.Tensor,  # (b, seq)
        labels,  # (b, word, attr) or None
        confidences,  # (b, word, attr) or None
        pooling_matrix,  # (b, word, seq)
        **_,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        batch_size, word_len, sequence_len = pooling_matrix.size()

        bert_out = self.bert(input_ids, attention_mask=(input_ids > 0))
        # create word-level representations using pooler matrix
        # (b, word, seq), (b, seq, hid) -> (b, word, hid)
        sequence_output = torch.bmm(pooling_matrix, bert_out.last_hidden_state)
        sequence_output = self.relu(self.dropout(sequence_output))  # (b, word, hid)

        # hiddens = [self.relu(layer(sequence_output)) for layer in self.output_layer]
        # (b, word, hid) -> (b, word, attr*hid) -> (b, word, attr, hid)
        output = self.output_layer(sequence_output).view(batch_size, word_len, self.num_attributes, -1)
        logits = self.classifier(self.relu(self.dropout(output)))  # (b, word, attr, 3)

        loss = None
        if labels is not None:
            raw_loss = F.cross_entropy(
                logits[:, :-1].permute(0, 3, 1, 2),
                labels,
                reduction="none",
                ignore_index=NerDataset.PAD_FOR_LABELS,
            )  # (b, word, attr)
            loss = (raw_loss * confidences).sum() / (confidences.sum() + 1e-6)

        return loss, logits
