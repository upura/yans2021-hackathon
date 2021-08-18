from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Union, Any, Optional

from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from dataclasses_json import dataclass_json

from util import is_chunk_end, is_chunk_start


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
class NerExample:
    tokens: list[str]
    word_idxs: tuple[list[int], dict[int, int]]
    labels: Optional[list[list[str]]]


def find_word_alignment(tokens):
    word_idxs = []
    sub2word = {}
    for idx, token in enumerate(tokens):
        if not token.startswith("##"):
            word_idxs.append(idx)
        sub2word[idx] = len(word_idxs) - 1

    # add word_idx for end offset
    if len(tokens) > 0:
        word_idxs.append(len(tokens))
        sub2word[len(tokens)] = len(word_idxs) - 1

    return word_idxs, sub2word


class ShinraData:
    def __init__(self, attributes: list[str], params: dict[str, Any]):
        self.attributes: list[str] = attributes
        self.attr2idx = {attr: idx for idx, attr in enumerate(self.attributes)}

        self.page_id = None
        self.page_title = None
        self.category = None
        self.plain_text = None
        self.tokens = None
        self.word_alignments = None
        self.sub2word = None
        self.text_offsets = None
        self.valid_line_ids = None
        self.nes = None

        for key, value in params.items():
            setattr(self, key, value)

    @classmethod
    def from_shinra2020_format(cls, input_path: Union[Path, str]):
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
                attributes: list[str] = [attr.strip() for attr in f if attr.strip() != ""]
        else:
            attributes_set = set()
            for page_id, anns in annotations.items():
                attributes_set |= {ann.attribute for ann in anns}
            with attributes_path.open(mode="wt") as f:
                f.write("\n".join(attributes_set) + "\n")

        docs = []
        for token_file in tqdm(
            tokens_dir.glob("*.txt"),
            # total=len([token for token in tokens_dir.glob("*.txt")]),
        ):
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
            word_alignments = [find_word_alignment(t) for t in tokens]
            sub2word = [w[1] for w in word_alignments]
            word_alignments = [w[0] for w in word_alignments]

            data = {
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
                data["nes"] = annotations[page_id]

            docs.append(cls(attributes, params=data))
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
    def _load_tokens(path: Path, vocab: list[str]):
        tokens: list[list[str]] = []
        text_offsets = []
        with path.open() as f:
            for line in f:
                raw_tokens = line.rstrip().split()
                line = [l.split(",") for l in raw_tokens]
                tokens.append([vocab[int(l[0])] for l in line])
                text_offsets.append([[l[1], l[2]] for l in line])

        return tokens, text_offsets

    # iobs = [sents1, sents2, ...]
    # sents1 = [[iob1_attr1, iob2_attr1, ...], [iob1_attr2, iob2_attr2, ...], ...]
    def add_nes_from_iob(self, iobs):
        assert len(iobs) == len(
            self.valid_line_ids
        ), f"{len(iobs)}, {len(self.valid_line_ids)}"
        self.nes = []

        for line_id, sent_iob in zip(self.valid_line_ids, iobs):
            word2subword = self.word_alignments[line_id]
            tokens = self.tokens[line_id]
            text_offsets = self.text_offsets[line_id]
            for iob, attr in zip(sent_iob, self.attributes):
                ne = {}
                iob = [0] + iob + [0]
                for token_idx in range(1, len(iob)):
                    if is_chunk_end(iob[token_idx - 1], iob[token_idx]):
                        assert ne != {}
                        # token_idxは本来のものから+2されているので，word2subwordはneの外のはじめのtoken_id
                        end_offset = (
                            len(tokens)
                            if token_idx - 1 >= len(word2subword)
                            else word2subword[token_idx - 1]
                        )
                        # end_offset = len(tokens) if token_idx >= len(word2subword) else word2subword[token_idx-1]
                        ne["token_offset"]["end"] = {
                            "line_id": line_id,
                            "offset": end_offset,
                        }
                        ne["token_offset"]["text"] = " ".join(
                            tokens[
                                ne["token_offset"]["start"]["offset"] : ne[
                                    "token_offset"
                                ]["end"]["offset"]
                            ]
                        )

                        ne["text_offset"]["end"] = {
                            "line_id": line_id,
                            "offset": text_offsets[end_offset - 1][1],
                        }
                        ne["page_id"] = self.page_id
                        ne["title"] = self.page_title

                        self.nes.append(ne)
                        ne = {}

                    if is_chunk_start(iob[token_idx - 1], iob[token_idx]):
                        ne["attribute"] = attr
                        ne["token_offset"] = {
                            "start": {
                                "line_id": line_id,
                                "offset": word2subword[token_idx - 1],
                            }
                        }
                        ne["text_offset"] = {
                            "start": {
                                "line_id": line_id,
                                "offset": text_offsets[word2subword[token_idx - 1]][0],
                            }
                        }

    @property
    def ner_inputs(self) -> list[NerExample]:
        outputs: list[NerExample] = []
        iobs = self.iob
        for idx in self.valid_line_ids:
            sent = NerExample(
                tokens=self.tokens[idx],
                word_idxs=self.word_alignments[idx],
                labels=iobs[idx] if self.nes is not None else None,
            )
            outputs.append(sent)

        # outputs["input_ids"] = self.tokens
        # outputs["word_idxs"] = self.word_alignments.copy()

        # if self.nes is not None:
        #     outputs["labels"] = self.iob
        # else:
        #     outputs["labels"] = [None for i in range(len(self.tokens))]

        return outputs

    @property
    def words(self):
        all_words = []
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
        iobs = [
            [["O" for _ in range(len(tokens) - 1)] for _ in range(len(self.attributes))]
            for tokens in self.word_alignments
        ]
        for ne in self.nes:
            if "token_offset" not in ne:
                continue
            start_line = int(ne["token_offset"]["start"]["line_id"])
            start_offset = int(ne["token_offset"]["start"]["offset"])

            end_line = int(ne["token_offset"]["end"]["line_id"])
            end_offset = int(ne["token_offset"]["end"]["offset"])

            # 文を跨いだentityは除外
            if start_line != end_line:
                continue

            # 正解となるsubwordを含むwordまでタグ付
            attr_idx = self.attr2idx[ne["attribute"]]
            ne_start = self.sub2word[start_line][start_offset]
            ne_end = self.sub2word[end_line][end_offset - 1] + 1

            for idx in range(ne_start, ne_end):
                iobs[start_line][attr_idx][idx] = "B" if idx == ne_start else "I"

        return iobs


class NerDataset(Dataset):
    label2id = {"O": 0, "B": 1, "I": 2}
    # datas = [{"tokens": , "word_idxs": , "labels": }, ...]

    def __init__(self, examples: list[NerExample], tokenizer):
        self.tokenizer = tokenizer
        self.examples: list[NerExample] = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        input_ids = ["[CLS]"] + self.examples[item].tokens[:510] + ["[SEP]"]
        input_ids = self.tokenizer.convert_tokens_to_ids(input_ids)
        word_idxs = [idx + 1 for idx in self.examples[item].word_idxs if idx <= 510]

        labels = self.examples[item].labels
        if labels is not None:
            # truncate label using zip(_, word_idxs[:-1]), word_idxs[-1] is not valid idx (for end offset)
            labels = [
                [self.label2id[lbl] for lbl, _ in zip(label, word_idxs[:-1])]
                for label in labels
            ]

        return input_ids, word_idxs, labels


def ner_collate_fn(batch):
    tokens, word_idxs, labels = list(zip(*batch))
    if labels[0] is not None:
        labels = [[label[idx] for label in labels] for idx in range(len(labels[0]))]

    return {"tokens": tokens, "word_idxs": word_idxs, "labels": labels}


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("cl-tohoku/bert-base-japanese")
    shinra_dataset = ShinraData.from_shinra2020_format(
        "/data1/ujiie/shinra/tohoku_bert/Event/Event_Other"
    )
    dataset = NerDataset(shinra_dataset[0].ner_inputs, tokenizer)
    print(dataset[0])
