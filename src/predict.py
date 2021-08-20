import argparse
import json
import os
from pathlib import Path
from typing import Tuple

import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

from dataset import ShinraData, NerDataset, ner_collate_fn
from model import BertForMultilabelNER, create_pooler_matrix

device = "cuda:0" if torch.cuda.is_available() else "cpu"


def ner_for_shinradata(model, tokenizer, shinra_dataset, device):
    ner_examples = shinra_dataset.to_ner_examples()
    dataset = NerDataset(ner_examples, tokenizer)
    total_preds, _ = predict(model, dataset, device, sent_wise=True)

    shinra_dataset.add_nes_from_iob(total_preds)

    return shinra_dataset


def predict(
    model: nn.Module,
    dataset: NerDataset,
    device,
    sent_wise=False
) -> Tuple[list, list]:
    model.eval()
    dataloader = DataLoader(dataset, batch_size=8, collate_fn=ner_collate_fn)

    total_preds = []
    total_trues = []
    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            input_ids = batch["input_ids"]  # (b, seq)
            word_idxs = batch["word_idxs"]  # (b, word)
            labels = batch["labels"]  # (b, seq, attr) or None

            attention_mask = input_ids > 0
            pooling_matrix = create_pooler_matrix(
                input_ids, word_idxs, pool_type="head"
            ).to(device)

            preds = model.predict(
                input_ids=input_ids,
                attention_mask=attention_mask,
                word_idxs=word_idxs,
                pooling_matrix=pooling_matrix,
            )

            total_preds.append(preds)
            # test dataの場合truesは使わないので適当にpredsを入れる
            total_trues.append(labels.permute(2, 0, 1) if labels is not None else preds)

    attr_num = len(total_preds[0])
    total_preds = [
        [pred for preds in total_preds for pred in preds[attr]]
        for attr in range(attr_num)
    ]
    total_trues = [
        [true for trues in total_trues for true in trues[attr]]
        for attr in range(attr_num)
    ]

    if sent_wise:
        total_preds = [
            [total_preds[attr][idx] for attr in range(attr_num)]
            for idx in range(len(total_preds[0]))
        ]
        total_trues = [
            [total_trues[attr][idx] for attr in range(attr_num)]
            for idx in range(len(total_trues[0]))
        ]

    return total_preds, total_trues


def parse_arg() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_path", type=str, help="Specify input path in SHINRA2020"
    )
    parser.add_argument(
        "--model_path", type=str, help="Specify attribute_list path in SHINRA2020"
    )
    parser.add_argument(
        "--output_path", type=str, help="Specify attribute_list path in SHINRA2020"
    )
    parser.add_argument(
        "--mode", type=str, choices=["leaderboard", "all"], default="all",
        help="Specify attribute_list path in SHINRA2020"
    )

    return parser.parse_args()


def main():
    args = parse_arg()

    bert = AutoModel.from_pretrained("cl-tohoku/bert-base-japanese")
    tokenizer = AutoTokenizer.from_pretrained("cl-tohoku/bert-base-japanese")

    category = str(args.input_path).split("/")[-1]

    # dataset = [ShinraData(), ....]
    dataset_cache_dir = Path(os.environ.get("SHINRA_CACHE_DIR", "../tmp"))
    dataset_cache_dir.mkdir(exist_ok=True)
    cache_path = dataset_cache_dir / f"{category}_{args.mode}_dataset.pkl"
    if cache_path.exists():
        print(f"Loading cached dataset from {cache_path}")
        shinra_dataset = joblib.load(cache_path)
    else:
        print(f"Cached dataset not found. Building one from {args.input_path}")
        shinra_dataset = ShinraData.from_shinra2020_format(Path(args.input_path), mode=args.mode)
        joblib.dump(shinra_dataset, cache_path, compress=3)

    model = BertForMultilabelNER(bert, len(shinra_dataset[0].attributes))
    model.load_state_dict(torch.load(args.model_path))
    model.to(device)

    dataset = [ner_for_shinradata(model, tokenizer, ds, device) for ds in shinra_dataset]
    with open(args.output_path, "w") as f:
        f.write(
            "\n".join(
                [json.dumps(ne, ensure_ascii=False) for d in dataset for ne in d.nes]
            )
        )


if __name__ == "__main__":
    main()
