from collections import defaultdict
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from dataclasses_json import dataclass_json
from tqdm import tqdm

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


@dataclass_json
@dataclass
class NEDataOffset:
    start: Optional[OffsetPoint]
    end: Optional[OffsetPoint]
    text: Optional[str]


@dataclass_json
@dataclass(frozen=True)
class NamedEntity:
    page_id: int  # int 型じゃないと submit に失敗する
    title: str
    attribute: str
    text_offset: NEDataOffset
    token_offset: NEDataOffset
    ENE: str

    @classmethod
    def from_annotation(cls, annotation: Annotation):
        return cls(
            int(annotation.page_id),
            annotation.title,
            annotation.attribute,
            text_offset=NEDataOffset(
                annotation.text_offset.start, annotation.text_offset.end, None
            ),
            token_offset=NEDataOffset(
                annotation.token_offset.start, annotation.token_offset.end, None
            ),
            ENE=annotation.ENE,
        )


class ShinraData:
    CATEGORY2ENE = {"City": "1.5.1.1", "Company": "1.4.6.2"}

    def __init__(self, attributes: List[str], params: Dict[str, Any]):
        self.attributes: List[str] = attributes
        self.attr2idx = {attr: idx for idx, attr in enumerate(self.attributes)}

        self.page_id: str = params["page_id"]
        self.page_title: str = params["page_title"]
        self.category: str = params["category"]
        self.ene: str = ShinraData.CATEGORY2ENE[self.category]
        self.tokens: List[List[str]] = params["tokens"]
        self.word_alignments: List[List[int]] = params["word_alignments"]
        self.sub2word: List[Dict[int, int]] = params["sub2word"]
        self.text_offsets: List[List[DataOffset]] = params["text_offsets"]
        self.valid_line_ids: List[int] = params["valid_line_ids"]
        self.nes: Optional[List[NamedEntity]] = (
            [NamedEntity.from_annotation(a) for a in params["nes"]]
            if "nes" in params
            else None
        )

    @classmethod
    def from_shinra2020_format(
        cls,
        input_path: Union[Path, str],
        mode: str = "all",  # "train", "leaderboard", "pseudo", or "all"
    ) -> List["ShinraData"]:
        input_path = Path(input_path)
        category = input_path.stem
        annotation_path = input_path / f"{category}_dist.json"
        vocab_path = input_path / "vocab.txt"
        attributes_path = input_path / "attributes.txt"
        tokens_dir = input_path / "tokenized" / mode

        annotations: Dict[str, List[Annotation]] = cls._load_annotation(annotation_path)
        vocab: List[str] = cls._load_vocab(vocab_path)

        # create attributes
        if attributes_path.exists():
            with attributes_path.open() as f:
                attributes: List[str] = [
                    attr.strip() for attr in f if attr.strip() != ""
                ]
        else:
            attributes: List[str] = list(
                {ann.attribute for anns in annotations.values() for ann in anns}
            )
            attributes_path.write_text("\n".join(attributes) + "\n")

        with Pool() as p:
            args = [
                (token_file, vocab, category, annotations, attributes)
                for token_file in tokens_dir.glob("*.txt")
            ]
            docs = p.starmap(cls._parse_token_file, tqdm(args, total=len(args)))
        return docs

    @classmethod
    def _parse_token_file(
        cls,
        token_file: Path,
        vocab: List[str],
        category: str,
        annotations: Dict[str, List[Annotation]],
        attributes: List[str],
    ) -> "ShinraData":
        page_id: str = token_file.stem
        tokens, text_offsets = cls._load_tokens(token_file, vocab)
        valid_line_ids: List[int] = [
            idx for idx, token in enumerate(tokens) if len(token) > 0
        ]

        # find title
        title_line: str = "".join(
            [t[2:] if t.startswith("##") else t for t in tokens[4]]
        )
        pos = title_line.find("-WikipediaDump")
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

    @staticmethod
    def _load_annotation(path: Path) -> Dict[str, List[Annotation]]:
        annotations: Dict[str, List[Annotation]] = defaultdict(list)
        with path.open() as f:
            for line in f:
                line = line.rstrip()
                if not line:
                    continue
                annotation: Annotation = Annotation.from_json(line)
                annotations[annotation.page_id].append(annotation)
        return annotations

    @staticmethod
    def _load_vocab(path: Path) -> List[str]:
        with path.open() as f:
            return [line.rstrip() for line in f if line.rstrip()]

    @staticmethod
    def _load_tokens(
        path: Path, vocab: List[str]
    ) -> Tuple[List[List[str]], List[List[DataOffset]]]:
        tokens_list: List[List[str]] = []
        text_offsets: List[List[DataOffset]] = []
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
    def _find_word_alignment(tokens: List[str]) -> Tuple[List[int], Dict[int, int]]:
        word_idxs: List[int] = []  # 単語先頭に相当するサブワードIDのリスト．長さは単語数と等しい
        sub2word: Dict[int, int] = {}
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
    def add_nes_from_iob(self, iobs: List[List[List[int]]]):
        assert len(iobs) == len(
            self.valid_line_ids
        ), f"{len(iobs)}, {len(self.valid_line_ids)}"
        self.nes: List[NamedEntity] = []

        for line_id, sent_iob in zip(self.valid_line_ids, iobs):
            word2subword: List[int] = self.word_alignments[line_id]
            tokens: List[str] = self.tokens[line_id]
            text_offsets: List[DataOffset] = self.text_offsets[line_id]
            for iob, attr in zip(sent_iob, self.attributes):
                text_offset = NEDataOffset(None, None, None)
                token_offset = NEDataOffset(None, None, None)
                iob: List[int] = [0] + iob + [0]
                for token_idx in range(1, len(iob)):
                    if is_chunk_end(iob[token_idx - 1], iob[token_idx]):
                        # token_idxは本来のものから+2されているので，word2subwordはneの外のはじめのtoken_id
                        end_offset = (
                            len(tokens)
                            if token_idx - 1 >= len(word2subword)
                            else word2subword[token_idx - 1]
                        )
                        token_offset.end = OffsetPoint(line_id, end_offset)
                        text_offset.end = text_offsets[end_offset - 1].end
                        text = "".join(
                            [t[2:] if t.startswith("##") else t
                             for t in tokens[token_offset.start.offset : token_offset.end.offset]]
                        )
                        token_offset.text = text
                        text_offset.text = text

                        assert text_offset.start and text_offset.end
                        assert token_offset.start and token_offset.end

                        self.nes.append(
                            NamedEntity(
                                page_id=int(self.page_id),
                                title=self.page_title,
                                attribute=attr,
                                text_offset=text_offset,
                                token_offset=token_offset,
                                ENE=self.ene,
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

    @property
    def words(self) -> List[List[str]]:
        all_words: List[List[str]] = []
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

    def to_iob(self) -> List[List[List[str]]]:
        """
        %%% IOB for ** only word-level iob2 tag **
        iobs = [sent, sent, ...]
        sent = [[Token1_attr1_iob, Token2_attr1_iob, ...], [Token1_attr2_iob, Token2_attr2_iob, ...], ...]

        {"O": 0, "B": 1, "I": 2}
        """
        iobs: List[List[List[str]]] = [
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
