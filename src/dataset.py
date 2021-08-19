from collections import defaultdict
from dataclasses import asdict, dataclass
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Optional, Union

import torch
import torch.nn.utils.rnn as rnn
from dataclasses_json import dataclass_json
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizer

from util import is_chunk_end, is_chunk_start


def parse_token_file(token_file, vocab, category, annotations, cls, attributes):
    page_id: str = token_file.stem
    tokens, text_offsets = cls._load_tokens(token_file, vocab)
    valid_line_ids: list[int] = [
        idx for idx, token in enumerate(tokens) if len(token) > 0
    ]

    # find title
    title_line: str = "".join([t[2:] if t.startswith("##") else t for t in tokens[4]])
    pos = title_line.find("-jawiki")
    title = title_line[:pos]

    # find word alignments = start positions of words
    word_alignments, sub2word = [], []
    for token in tokens:
        word_idxs, s2w = cls._find_word_alignment(token)
        word_alignments.append(word_idxs)
        sub2word.append(s2w)

    params = {
        "page_id": page_id,
        "page_title": title,
        "category": category,
        "tokens": tokens,
        "text_offsets": text_offsets,
        "word_alignments": word_alignments,
        "sub2word": sub2word,
        "valid_line_ids": valid_line_ids,
    }

    if page_id in annotations:
        params["nes"] = annotations[page_id]

    return cls(attributes, params=params)


@dataclass_json
@dataclass(frozen=True)
class OffsetPoint:
    line_id: int
    offset: int


@dataclass_json
@dataclass(frozen=True)
class DataOffset:
    start: OffsetPoint
    end: OffsetPoint


@dataclass_json
@dataclass(frozen=True)
class Annotation:
    page_id: str
    title: str
    attribute: str
    html_offset: DataOffset
    text_offset: DataOffset
    token_offset: DataOffset
    ENE: str


@dataclass
class NEDataOffset:
    start: Optional[OffsetPoint]
    end: Optional[OffsetPoint]
    text: Optional[str]


@dataclass(frozen=True)
class NamedEntity:
    page_id: str
    title: str
    attribute: str
    text_offset: NEDataOffset
    token_offset: NEDataOffset

    @classmethod
    def from_annotation(cls, annotation: Annotation):
        return cls(
            annotation.page_id,
            annotation.title,
            annotation.attribute,
            text_offset=NEDataOffset(
                annotation.text_offset.start, annotation.text_offset.end, None
            ),
            token_offset=NEDataOffset(
                annotation.token_offset.start, annotation.token_offset.end, None
            ),
        )


@dataclass
class NerExample:
    tokens: list[str]
    word_idxs: list[int]
    labels: Optional[list[list[str]]]


class ShinraData:
    def __init__(self, attributes: list[str], params: dict[str, Any]):
        self.attributes: list[str] = attributes
        self.attr2idx = {attr: idx for idx, attr in enumerate(self.attributes)}

        self.page_id: str = params["page_id"]
        self.page_title: str = params["page_title"]
        self.category: str = params["category"]
        self.tokens: list[list[str]] = params["tokens"]
        self.word_alignments: list[list[int]] = params["word_alignments"]
        self.sub2word: list[dict[int, int]] = params["sub2word"]
        self.text_offsets: list[list[DataOffset]] = params["text_offsets"]
        self.valid_line_ids: list[int] = params["valid_line_ids"]
        self.nes: Optional[list[NamedEntity]] = (
            [NamedEntity.from_annotation(a) for a in params["nes"]]
            if "nes" in params
            else None
        )

    @classmethod
    def from_shinra2020_format(cls, input_path: Union[Path, str]) -> list["ShinraData"]:
        input_path = Path(input_path)
        category = input_path.stem
        annotation_path = input_path / f"{category}_dist.json"
        vocab_path = input_path / "vocab.txt"
        attributes_path = input_path / "attributes.txt"
        tokens_dir = input_path / "tokens"

        annotations: dict[str, list[Annotation]] = cls._load_annotation(annotation_path)
        vocab: list[str] = cls._load_vocab(vocab_path)

        # create attributes
        if attributes_path.exists():
            with attributes_path.open() as f:
                attributes: list[str] = [
                    attr.strip() for attr in f if attr.strip() != ""
                ]
        else:
            attributes: list[str] = list(
                {ann.attribute for anns in annotations.values() for ann in anns}
            )
            attributes_path.write_text("\n".join(attributes) + "\n")

        with Pool() as p:
            args = [
                (token_file, vocab, category, annotations, cls, attributes)
                for token_file in tokens_dir.glob("*.txt")
            ]
            docs = p.starmap(parse_token_file, tqdm(args, total=len(args)))
        return docs

    @staticmethod
    def _load_annotation(path: Path) -> dict[str, list[Annotation]]:
        annotations: dict[str, list[Annotation]] = defaultdict(list)
        with path.open() as f:
            for line in f:
                line = line.rstrip()
                if not line:
                    continue
                annotation: Annotation = Annotation.from_json(line)
                annotations[annotation.page_id].append(annotation)
        return annotations

    @staticmethod
    def _load_vocab(path: Path) -> list[str]:
        with path.open() as f:
            return [line.rstrip() for line in f if line.rstrip()]

    @staticmethod
    def _load_tokens(
        path: Path, vocab: list[str]
    ) -> tuple[list[list[str]], list[list[DataOffset]]]:
        tokens_list: list[list[str]] = []
        text_offsets: list[list[DataOffset]] = []
        with path.open() as f:
            for line_id, line in enumerate(f):
                tokens, offsets = [], []
                for raw_token in line.rstrip().split():
                    token_id, start_idx, end_idx = raw_token.split(",")
                    tokens.append(vocab[int(token_id)])
                    offsets.append(
                        DataOffset(
                            OffsetPoint(line_id, int(start_idx)),
                            OffsetPoint(line_id, int(end_idx)),
                        )
                    )
                tokens_list.append(tokens)
                text_offsets.append(offsets)

        return tokens_list, text_offsets

    @staticmethod
    def _find_word_alignment(tokens: list[str]) -> tuple[list[int], dict[int, int]]:
        word_idxs: list[int] = []  # 単語先頭に相当するサブワードIDのリスト．長さは単語数と等しい
        sub2word: dict[int, int] = {}
        for idx, token in enumerate(tokens):
            if not token.startswith("##"):
                word_idxs.append(idx)
            sub2word[idx] = len(word_idxs) - 1

        # add word_idx for end offset
        if len(tokens) > 0:
            word_idxs.append(len(tokens))
            sub2word[len(tokens)] = len(word_idxs) - 1

        return word_idxs, sub2word

    # iobs = [sents1, sents2, ...]
    # sents1 = [[iob1_attr1, iob2_attr1, ...], [iob1_attr2, iob2_attr2, ...], ...]
    def add_nes_from_iob(self, iobs: list[list[list[int]]]):
        assert len(iobs) == len(
            self.valid_line_ids
        ), f"{len(iobs)}, {len(self.valid_line_ids)}"
        self.nes: list[NamedEntity] = []

        for line_id, sent_iob in zip(self.valid_line_ids, iobs):
            word2subword: list[int] = self.word_alignments[line_id]
            tokens: list[str] = self.tokens[line_id]
            text_offsets: list[DataOffset] = self.text_offsets[line_id]
            for iob, attr in zip(sent_iob, self.attributes):
                text_offset = NEDataOffset(None, None, None)
                token_offset = NEDataOffset(None, None, None)
                iob: list[int] = [0] + iob + [0]
                for token_idx in range(1, len(iob)):
                    if is_chunk_end(iob[token_idx - 1], iob[token_idx]):
                        # token_idxは本来のものから+2されているので，word2subwordはneの外のはじめのtoken_id
                        end_offset = (
                            len(tokens)
                            if token_idx - 1 >= len(word2subword)
                            else word2subword[token_idx - 1]
                        )
                        token_offset.end = OffsetPoint(line_id, end_offset)
                        token_offset.text = " ".join(
                            tokens[token_offset.start.offset : token_offset.end.offset]
                        )
                        text_offset.end = text_offsets[end_offset - 1].end

                        assert text_offset.start and text_offset.end
                        assert token_offset.start and token_offset.end

                        self.nes.append(
                            NamedEntity(
                                page_id=self.page_id,
                                title=self.page_title,
                                attribute=attr,
                                text_offset=text_offset,
                                token_offset=token_offset,
                            )
                        )

                    if is_chunk_start(iob[token_idx - 1], iob[token_idx]):
                        text_offset = NEDataOffset(None, None, None)
                        token_offset = NEDataOffset(None, None, None)
                        token_offset.start = OffsetPoint(
                            line_id, word2subword[token_idx - 1]
                        )
                        text_offset.start = text_offsets[
                            word2subword[token_idx - 1]
                        ].start

    def to_ner_examples(self) -> list[NerExample]:
        outputs: list[NerExample] = []
        iobs = self.iob
        for idx in self.valid_line_ids:
            sent = NerExample(
                tokens=self.tokens[idx],
                word_idxs=self.word_alignments[idx],
                labels=iobs[idx] if self.nes is not None else None,
            )
            outputs.append(sent)
        return outputs

    @property
    def words(self) -> list[list[str]]:
        all_words: list[list[str]] = []
        for tokens, word_alignments in zip(self.tokens, self.word_alignments):
            words = []
            prev_idx = 0
            for idx in word_alignments[1:] + [-1]:
                inword_subwords = tokens[prev_idx:idx]
                inword_subwords = [
                    s[2:] if s.startswith("##") else s for s in inword_subwords
                ]
                words.append("".join(inword_subwords))
                prev_idx = idx
            all_words.append(words)
        return all_words

    @property
    def iob(self) -> list[list[list[str]]]:
        """
        %%% IOB for ** only word-level iob2 tag **
        iobs = [sent, sent, ...]
        sent = [[Token1_attr1_iob, Token2_attr1_iob, ...], [Token1_attr2_iob, Token2_attr2_iob, ...], ...]

        {"O": 0, "B": 1, "I": 2}
        """
        iobs: list[list[list[str]]] = [
            [["O"] * (len(tokens) - 1) for _ in self.attributes]
            for tokens in self.word_alignments
        ]
        for ne in self.nes:
            start_line: int = ne.token_offset.start.line_id
            start_offset: int = ne.token_offset.start.offset
            end_line: int = ne.token_offset.end.line_id
            end_offset: int = ne.token_offset.end.offset

            # 文を跨いだentityは除外
            if start_line != end_line:
                continue

            # 正解となるsubwordを含むwordまでタグ付
            attr_idx: int = self.attr2idx[ne.attribute]
            ne_start: int = self.sub2word[start_line][start_offset]
            ne_end: int = self.sub2word[end_line][end_offset - 1] + 1

            for idx in range(ne_start, ne_end):
                iobs[start_line][attr_idx][idx] = "B" if idx == ne_start else "I"

        return iobs


@dataclass(frozen=True)
class InputFeature:
    input_ids: list[int]
    word_idxs: list[int]
    labels: Optional[list[list[int]]]


class NerDataset(Dataset):
    LABEL2ID = {"O": 0, "B": 1, "I": 2}
    MAX_SEQ_LENGTH = 512
    PAD_FOR_INPUT_IDS = 0
    PAD_FOR_LABELS = -1

    def __init__(self, examples: list[NerExample], tokenizer: PreTrainedTokenizer):
        self.tokenizer: PreTrainedTokenizer = tokenizer
        assert tokenizer.pad_token_id == self.PAD_FOR_INPUT_IDS
        self.examples: list[NerExample] = examples

    @staticmethod
    def _convert_example_to_feature(
        example: NerExample, tokenizer: PreTrainedTokenizer
    ) -> InputFeature:
        input_tokens: list[str] = (
            ["[CLS]"] + example.tokens[: NerDataset.MAX_SEQ_LENGTH - 2] + ["[SEP]"]
        )
        input_ids: list[int] = tokenizer.convert_tokens_to_ids(input_tokens)
        word_idxs = [
            idx + 1 for idx in example.word_idxs if idx <= NerDataset.MAX_SEQ_LENGTH - 2
        ]

        labels = example.labels
        if labels is not None:
            # truncate label using zip(_, word_idxs[:-1]), word_idxs[-1] is not valid idx (for end offset)
            labels = [
                [NerDataset.LABEL2ID[lbl] for lbl, _ in zip(label, word_idxs[:-1])]
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

    def __getitem__(self, idx: int) -> dict[str, Any]:
        feature = self._convert_example_to_feature(self.examples[idx], self.tokenizer)
        return asdict(feature)


def ner_collate_fn(features: list[dict[str, Any]]) -> dict[str, Any]:
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
                )  # (b, seq, attr)
                batch[field] = feats
    return batch


if __name__ == "__main__":
    _tokenizer = AutoTokenizer.from_pretrained("cl-tohoku/bert-base-japanese")
    shinra_dataset = ShinraData.from_shinra2020_format(
        "/data1/ujiie/shinra/tohoku_bert/Event/Event_Other"
    )
    dataset = NerDataset(shinra_dataset[0].to_ner_examples(), _tokenizer)
    print(dataset[0])
