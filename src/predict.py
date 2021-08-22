import argparse
import os
from collections import OrderedDict
from pathlib import Path
from typing import Tuple, List

import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizer

from dataset import ShinraData, NerDataset, ner_collate_fn
from model import BertForMultilabelNER, create_pooler_matrix


def ner_for_shinradata(
    model: nn.Module,
    tokenizer: PreTrainedTokenizer,
    shinra_data: ShinraData
) -> ShinraData:
    assert shinra_data.nes is None
    dataset = NerDataset.from_shinra(shinra_data, tokenizer)
    total_preds, _ = predict(model, dataset, sent_wise=True)
    shinra_data.add_nes_from_iob(total_preds)
    return shinra_data


def predict(
    model: nn.Module,
    dataset: NerDataset,
    sent_wise: bool = False
) -> Tuple[List[List[List[int]]], List[List[List[int]]]]:

    dataloader = DataLoader(dataset, batch_size=8, collate_fn=ner_collate_fn)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model = torch.nn.DataParallel(model)
    model.eval()

    total_preds: List[List[List[List[int]]]] = []
    total_trues: List[List[List[List[int]]]] = []
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

            # (b, seq, attr, 3)
            _, logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pooling_matrix=pooling_matrix,
            )
            preds: List[List[List[int]]] = []  # (attr, b, seq)
            # attribute loop
            for attr_idx in range(logits.size(2)):
                attr_labels: List[List[int]] = viterbi(logits[:, :, attr_idx, :].detach().cpu())  # (b, seq)
                preds.append([
                    label[: len(word_idx) - 1]
                    for label, word_idx in zip(attr_labels, word_idxs)
                ])

            total_preds.append(preds)
            # test dataの場合truesは使わないので適当にpredsを入れる
            total_trues.append(labels.permute(2, 0, 1).tolist() if labels is not None else preds)

    num_attr: int = len(total_preds[0])
    total_preds_reshaped = [
        [pred for preds in total_preds for pred in preds[attr]]
        for attr in range(num_attr)
    ]  # (attr, N, seq)
    total_trues_reshaped = [
        [
            [t for t in true if t != NerDataset.PAD_FOR_LABELS]
            for trues in total_trues for true in trues[attr]
        ]
        for attr in range(num_attr)
    ]  # (attr, N, seq)

    if sent_wise:
        total_preds_reshaped = [
            [total_preds_reshaped[attr][idx] for attr in range(num_attr)]
            for idx in range(len(total_preds_reshaped[0]))
        ]
        total_trues_reshaped = [
            [total_trues_reshaped[attr][idx] for attr in range(num_attr)]
            for idx in range(len(total_trues_reshaped[0]))
        ]

    return total_preds_reshaped, total_trues_reshaped


def viterbi(logits, penalty=float("inf")) -> List[List[int]]:
    num_tags = 3

    # 0: O, 1: B, 2: I
    penalties = torch.zeros((num_tags, num_tags))
    penalties[0][2] = penalty

    all_preds: List[List[int]] = []
    for logit in logits:
        pred_tags: List[int] = [0]
        for l in logit:
            transit_penalty = penalties[pred_tags[-1]]
            l = l - transit_penalty
            tag = torch.argmax(l, dim=-1)
            pred_tags.append(tag.item())
        all_preds.append(pred_tags[1:])
    return all_preds


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
        help="Specify 'leaderboard' to evaluate leaderboard data and specify 'all' to evaluate all data"
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
        shinra_datum = joblib.load(cache_path)
    else:
        print(f"Cached dataset not found. Building one from {args.input_path}")
        shinra_datum = ShinraData.from_shinra2020_format(Path(args.input_path), mode=args.mode)
        joblib.dump(shinra_datum, cache_path, compress=3)

    model = BertForMultilabelNER(bert, len(shinra_datum[0].attributes))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(OrderedDict({k.replace('module.', ''): v for k, v in state_dict.items()}))

    # dataset = [ner_for_shinradata(model, tokenizer, ds) for ds in shinra_dataset]
    with open(args.output_path, "w") as f:
        for data in shinra_datum:
            processed_data = ner_for_shinradata(model, tokenizer, data)
            if processed_data.nes:
                f.write("\n".join(ne.to_json(ensure_ascii=False) for ne in processed_data.nes) + "\n")


if __name__ == "__main__":
    main()
