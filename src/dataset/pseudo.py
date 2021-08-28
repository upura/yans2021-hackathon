from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Generator, Tuple

from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from dataclasses_json import dataclass_json

from .shinra import ShinraData, DataOffset


@dataclass(frozen=True)
@dataclass_json
class AttributeResult:
    html_offset: DataOffset
    text_offset: DataOffset
    system: List[str]


@dataclass(frozen=True)
@dataclass_json
class SystemResult:
    title: str
    page_id: str
    result: Dict[str, List[AttributeResult]]

    def to_iob_conf(self, shinra_data: ShinraData) -> Tuple[List[List[List[str]]], List[List[List[float]]]]:
        """
        %%% IOB for ** only word-level iob2 tag **
        iobs = [sent, sent, ...]
        sent = [[Token1_attr1_iob, Token2_attr1_iob, ...], [Token1_attr2_iob, Token2_attr2_iob, ...], ...]

        {"O": 0, "B": 1, "I": 2}
        """
        iobs: List[List[List[str]]] = [
            [["O"] * (len(tokens) - 1) for _ in shinra_data.attributes]
            for tokens in shinra_data.word_alignments
        ]  # (sent, attr, word)
        # https://www.anlp.jp/proceedings/annual_meeting/2020/pdf_dir/F4-2.pdf, 表2
        CATEGORY_TO_NUM_SYSTEMS = {"Company": 7, "City": 6}
        num_systems = CATEGORY_TO_NUM_SYSTEMS[shinra_data.category]
        confs: List[List[List[float]]] = [
            [[1.] * (len(tokens) - 1) for _ in shinra_data.attributes]
            for tokens in shinra_data.word_alignments
        ]  # (sent, attr, word)
        for attr, attr_results in self.result.items():
            attr_idx: int = shinra_data.attr2idx[attr]
            for attr_result in attr_results:
                start_line: int = attr_result.text_offset.start.line_id
                start_offset: int = attr_result.text_offset.start.offset
                end_line: int = attr_result.text_offset.end.line_id
                end_offset: int = attr_result.text_offset.end.offset

                # 文を跨いだentityは除外
                if start_line != end_line:
                    continue

                for idx in range(start_offset, end_offset + 1):
                    iobs[start_line][attr_idx][idx] = "B" if idx == start_offset else "I"
                    confs[start_line][attr_idx][idx] = len(attr_result.system) / num_systems  # normalize

        return iobs, confs


@dataclass
class PseudoExample:
    tokens: List[str]
    word_idxs: List[int]
    labels: List[List[str]]
    confidence: List[List[float]]


@dataclass(frozen=True)
class InputFeature:
    input_ids: List[int]
    word_idxs: List[int]
    labels: Optional[List[List[int]]]


class PseudoDataset(Dataset):
    LABEL2ID = {"O": 0, "B": 1, "I": 2}
    MAX_SEQ_LENGTH = 512
    PAD_FOR_INPUT_IDS = 0
    PAD_FOR_LABELS = -1

    def __init__(self, examples: List[PseudoExample], tokenizer: PreTrainedTokenizer):
        self.tokenizer: PreTrainedTokenizer = tokenizer
        assert tokenizer.pad_token_id == self.PAD_FOR_INPUT_IDS
        self.examples: List[PseudoExample] = examples

    @classmethod
    def from_shinra(
        cls,
        shinra_data: List[ShinraData],
        system_result_path: Union[str, Path],
        tokenizer: PreTrainedTokenizer,
    ) -> "PseudoDataset":
        results: Dict[str, SystemResult] = {}
        with Path(system_result_path).open() as f:
            for line in f:
                system_result: SystemResult = SystemResult.from_json(line.strip())
                results[system_result.page_id] = system_result
        examples = [ex for data in shinra_data for ex in cls._shinra2examples(data, results[data.page_id])]
        return cls(examples, tokenizer)

    @staticmethod
    def _shinra2examples(shinra_data: ShinraData, result: SystemResult) -> Generator[PseudoExample, None, None]:
        iobs: List[List[List[str]]]  # (sent, attr, word)
        confs: List[List[List[float]]]  # (sent, attr, word)
        iobs, confs = result.to_iob_conf(shinra_data)
        for idx in shinra_data.valid_line_ids:
            example = PseudoExample(
                tokens=shinra_data.tokens[idx],
                word_idxs=shinra_data.word_alignments[idx],
                labels=iobs[idx],
                confidence=confs[idx],
            )
            yield example

    @staticmethod
    def _convert_example_to_feature(
        example: PseudoExample, tokenizer: PreTrainedTokenizer
    ) -> InputFeature:
        input_tokens: List[str] = (
            ["[CLS]"] + example.tokens[: PseudoDataset.MAX_SEQ_LENGTH - 2] + ["[SEP]"]
        )
        input_ids: List[int] = tokenizer.convert_tokens_to_ids(input_tokens)
        word_idxs = [
            idx + 1 for idx in example.word_idxs if idx <= PseudoDataset.MAX_SEQ_LENGTH - 2
        ]

        labels = example.labels
        if labels is not None:
            # truncate label using zip(_, word_idxs[:-1]), word_idxs[-1] is not valid idx (for end offset)
            labels = [
                [PseudoDataset.LABEL2ID[lbl] for lbl, _ in zip(label, word_idxs[:-1])]
                for label in labels
            ]

        feature = InputFeature(
            input_ids=input_ids,  # (seq)
            word_idxs=word_idxs,  # (word)
            labels=labels,  # (attr, seq)
        )

        return feature

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        feature = self._convert_example_to_feature(self.examples[idx], self.tokenizer)
        return asdict(feature)
