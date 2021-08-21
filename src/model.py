from typing import List

import torch
import torch.nn as nn
from transformers import BertModel


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

        # classifier that classifies token into IOB tag (B, I, O) for each attribute
        output_layer = [nn.Linear(768, 768) for _ in range(attribute_num)]
        self.output_layer = nn.ModuleList(output_layer)

        self.relu = nn.ReLU()

        # classifier that classifies token into IOB tag (B, I, O) for each attribute
        classifiers = [nn.Linear(768, 3) for _ in range(attribute_num)]
        self.classifiers = nn.ModuleList(classifiers)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        pooling_matrix=None,
    ):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]  # (b, seq, hid)
        # create word-level representations using pooler matrix
        sequence_output = torch.bmm(pooling_matrix, sequence_output)  # (b, seq, hid)
        sequence_output = self.dropout(sequence_output)  # (b, seq, hid)

        # hiddens = [self.relu(layer(sequence_output)) for layer in self.output_layer]
        # logits = [classifier(hiddens) for classifier, hiddens in zip(self.classifiers, hiddens)]
        logits = [
            classifier(sequence_output) for classifier in self.classifiers
        ]  # (attr, b, seq, 3)

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
            loss = 0
            labels = labels.permute(2, 0, 1).contiguous()  # (attr, b, seq)
            for label, logit in zip(labels, logits):
                loss += loss_fn(logit[:, :-1, :].reshape(-1, 3), label.view(-1)) / len(labels)

        output = (logits,) + outputs[2:]
        return loss, output

    @staticmethod
    def viterbi(logits, penalty=float("inf")):
        num_tags = 3

        # 0: O, 1: B, 2: I
        penalties = torch.zeros((num_tags, num_tags))
        penalties[0][2] = penalty

        all_preds = []
        for logit in logits:
            pred_tags = [0]
            for l in logit:
                transit_penalty = penalties[pred_tags[-1]]
                l = l - transit_penalty
                tag = torch.argmax(l, dim=-1)
                pred_tags.append(tag.item())
            all_preds.append(pred_tags[1:])
        return all_preds
