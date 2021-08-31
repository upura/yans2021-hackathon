import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import List

import _pickle as pickle
import torch.nn as nn
from seqeval.metrics import f1_score

from dataset.ner import NerDataset
from dataset.shinra import ShinraData
from predict import predict
from util import decode_iob


def parse_arg() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_path", type=str, help="Specify input path in SHINRA2020"
    )
    parser.add_argument(
        "--save_path", type=str, help="Specify path to directory where trained checkpoints are saved"
    )
    parser.add_argument(
        "--additional_name", "-a", type=str, help="Specify any string to identify experiment condition"
    )

    return parser.parse_args()


def evaluate(model: nn.Module, dataset: NerDataset, attributes: List[str]):
    total_preds, total_trues = predict(model, dataset, sent_wise=False)
    total_preds = decode_iob(total_preds, attributes)
    total_trues = decode_iob(total_trues, attributes)

    f1 = f1_score(total_trues, total_preds)
    return f1


def load_shinra_datum(input_path: Path, category: str, mode: str) -> List[ShinraData]:
    dataset_cache_dir = Path(os.environ.get("SHINRA_CACHE_DIR", "../tmp"))
    dataset_cache_dir.mkdir(exist_ok=True)
    cache_path = dataset_cache_dir / f"{category}_{mode}_dataset.pkl"
    if cache_path.exists():
        print(f"Loading cached dataset from {cache_path}")
        with cache_path.open(mode="rb") as f:
            shinra_datum = pickle.load(f)
    else:
        print(f"Cached shinra_datum not found. Building one from {input_path}")
        shinra_datum = ShinraData.from_shinra2020_format(input_path, mode=mode)
        with cache_path.open(mode="wb") as f:
            pickle.dump(shinra_datum, f)
    return shinra_datum


def export(
    shinra_datum: List[ShinraData],
    attributes: List[str],
    save_dir: Path,
    args: argparse.Namespace,
):
    all_words = []
    all_labels = []
    for shinra_data in shinra_datum:
        if not shinra_data.nes:
            continue
        # words: List[List[str]] = shinra_data.words  # (sent, word)
        # iobs = shinra_data.to_iob()  # (sent, attr, word)
        for words, iobs in zip(shinra_data.words, shinra_data.to_iob()):
            words: List[str]  # (word)
            iobs: List[List[str]]  # (attr, word)
            labels: List[str] = ["O"] * len(words)
            for attr_idx, iob in enumerate(iobs):
                for i, lbl in enumerate(iob):
                    if lbl in ("I", "B"):
                        labels[i] = f"{lbl}-{attributes[attr_idx]}"
            filtered_words = []
            filtered_labels = []
            for word, label in zip(words, labels):
                if word == '':
                    continue
                filtered_words.append(word)
                filtered_labels.append(label)
            if filtered_words:
                all_words.append(filtered_words)
                all_labels.append(filtered_labels)

    with open('tokens.txt', 'wb') as f:
        pickle.dump(all_words, f)

    with open('labels.txt', 'wb') as f:
        pickle.dump(all_labels, f)


def main():
    args = parse_arg()

    # tokenizer = AutoTokenizer.from_pretrained(args.bert)

    input_path = Path(args.input_path)
    category = input_path.parts[-1]
    shinra_datum = load_shinra_datum(input_path, category, mode="train")

    # train_shinra_datum, valid_shinra_datum = train_test_split(shinra_datum, test_size=0.1)
    # train_dataset = NerDataset.from_shinra(train_shinra_datum, tokenizer)
    # valid_dataset = NerDataset.from_shinra(valid_shinra_datum, tokenizer)

    save_dir = Path(args.save_path).joinpath(
        f"{category}{datetime.now().strftime(r'%m%d_%H%M')}" +
        (f"_{args.additional_name}" if args.additional_name else "")
    )
    save_dir.mkdir(parents=True)

    export(shinra_datum, shinra_datum[0].attributes, save_dir, args)


if __name__ == "__main__":
    main()
