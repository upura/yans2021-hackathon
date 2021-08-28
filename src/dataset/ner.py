from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Union, Generator

import torch
import torch.nn.utils.rnn as rnn
from torch.utils.data import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizer

from .shinra import ShinraData


@dataclass
class NerExample:
    tokens: List[str]
    word_idxs: List[int]
    labels: Optional[List[List[str]]]


@dataclass(frozen=True)
class InputFeature:
    input_ids: List[int]
    word_idxs: List[int]
    labels: Optional[List[List[int]]]
    confidences: List[List[float]]


class NerDataset(Dataset):
    LABEL2ID = {"O": 0, "B": 1, "I": 2}
    MAX_SEQ_LENGTH = 256
    PAD_FOR_INPUT_IDS = 0
    PAD_FOR_LABELS = -1

    def __init__(self, examples: List[NerExample], tokenizer: PreTrainedTokenizer):
        self.tokenizer: PreTrainedTokenizer = tokenizer
        assert tokenizer.pad_token_id == self.PAD_FOR_INPUT_IDS
        self.examples: List[NerExample] = examples

    @classmethod
    def from_shinra(
        cls,
        shinra_data: Union[ShinraData, List[ShinraData]],
        tokenizer: PreTrainedTokenizer
    ) -> "NerDataset":
        if isinstance(shinra_data, list):
            examples = [ex for data in shinra_data for ex in cls._shinra2examples(data)]
        else:
            examples = list(cls._shinra2examples(shinra_data))
        return cls(examples, tokenizer)

    @staticmethod
    def _shinra2examples(shinra_data: ShinraData) -> Generator[NerExample, None, None]:
        iobs = shinra_data.to_iob() if shinra_data.nes is not None else None
        for idx in shinra_data.valid_line_ids:
            example = NerExample(
                tokens=shinra_data.tokens[idx],
                word_idxs=shinra_data.word_alignments[idx],
                labels=iobs[idx] if shinra_data.nes is not None else None,
            )
            yield example

    @staticmethod
    def _convert_example_to_feature(
        example: NerExample, tokenizer: PreTrainedTokenizer
    ) -> InputFeature:
        input_tokens: List[str] = (
            ["[CLS]"] + example.tokens[: NerDataset.MAX_SEQ_LENGTH - 2] + ["[SEP]"]
        )
        input_ids: List[int] = tokenizer.convert_tokens_to_ids(input_tokens)
        word_idxs = [
            idx + 1 for idx in example.word_idxs if idx <= NerDataset.MAX_SEQ_LENGTH - 2
        ]

        labels = confs = None
        if example.labels is not None:
            # truncate label using zip(_, word_idxs[:-1]), word_idxs[-1] is not valid idx (for end offset)
            labels = [
                [NerDataset.LABEL2ID[lbl] for lbl, _ in zip(label, word_idxs[:-1])]
                for label in example.labels
            ]
            confs = [
                [1. for _ in zip(label, word_idxs[:-1])]
                for label in example.labels
            ]

        feature = InputFeature(
            input_ids=input_ids,  # (seq)
            word_idxs=word_idxs,  # (word)
            labels=labels,  # (attr, word)
            confidences=confs,  # (attr, word)
        )

        return feature

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        feature = self._convert_example_to_feature(self.examples[idx], self.tokenizer)
        return asdict(feature)


def ner_collate_fn(features: List[Dict[str, Any]]) -> Dict[str, Any]:
    first: dict = features[0]
    batch = {}
    for field in first.keys():
        if field == "input_ids":
            feats = rnn.pad_sequence(
                [torch.as_tensor(f[field]) for f in features],
                batch_first=True,
                padding_value=NerDataset.PAD_FOR_INPUT_IDS,
            )  # (b, seq)
            batch[field] = feats
        elif field == "word_idxs":
            batch[field] = [f[field] for f in features]
        elif field == "labels":
            batch[field] = None
            if first[field] is not None:
                feats = rnn.pad_sequence(
                    [torch.as_tensor(f[field]).transpose(0, 1) for f in features],
                    batch_first=True,
                    padding_value=NerDataset.PAD_FOR_LABELS,
                )  # (b, word, attr)
                batch[field] = feats
        elif field == "confidences":
            batch[field] = None
            if first[field] is not None:
                feats = rnn.pad_sequence(
                    [torch.as_tensor(f[field]).transpose(0, 1) for f in features],
                    batch_first=True,
                    padding_value=0.0,
                )  # (b, word, attr)
                batch[field] = feats
    return batch


if __name__ == "__main__":
    _tokenizer = AutoTokenizer.from_pretrained("cl-tohoku/bert-base-japanese")
    shinra_dataset = ShinraData.from_shinra2020_format(
        "/data1/ujiie/shinra/tohoku_bert/Event/Event_Other"
    )
    dataset = NerDataset.from_shinra(shinra_dataset[0], _tokenizer)
    print(dataset[0])
