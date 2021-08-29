import argparse
import os
from collections import OrderedDict
from pathlib import Path
from typing import Tuple, List
import _pickle as pickle
from multiprocessing import Pool

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizer

from dataset.shinra import ShinraData
from dataset.ner import NerDataset, ner_collate_fn
from model import BertForMultilabelNER, create_pooler_matrix


def ner_for_shinradata(
    model: nn.Module,
    tokenizer: PreTrainedTokenizer,
    shinra_batch: List[ShinraData]
) -> List[ShinraData]:
    assert all(shinra_data.nes is None for shinra_data in shinra_batch)
    dataset = NerDataset.from_shinra(shinra_batch, tokenizer)
    total_preds, _ = predict(model, dataset, sent_wise=True)
    sidx = 0
    for shinra_data in shinra_batch:
        eidx = sidx + len(shinra_data.valid_line_ids)
        shinra_data.add_nes_from_iob(total_preds[sidx:eidx])
        sidx = eidx
    assert sidx == len(total_preds)
    return shinra_batch


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


def get_pred(logits: torch.Tensor, word_idxs) -> List[List[int]]:
    return [
        viterbi(logit)[: len(word_idx) - 1]
        for logit, word_idx in zip(logits, word_idxs)
    ]


def predict(
    model: nn.Module,
    dataset: NerDataset,
    sent_wise: bool = False
) -> Tuple[List[List[List[int]]], List[List[List[int]]]]:
    batch_size_per_gpu = 1536
    num_gpus = torch.cuda.device_count()
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size_per_gpu * max(1, num_gpus),
        collate_fn=ner_collate_fn,
        num_workers=4,
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()

    total_preds: List[List[List[List[int]]]] = []
    total_trues: List[List[List[List[int]]]] = []
    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }

            pooling_matrix = create_pooler_matrix(
                batch["input_ids"], batch["word_idxs"], pool_type="head"
            ).to(device)  # (b, word, seq)

            _, logits = model(**batch, pooling_matrix=pooling_matrix)  # _, (b, word, attr, 3)

            with Pool(processes=logits.size(2)) as p:
                args = [(logits[:, :, attr_idx, :].detach().cpu(), batch["word_idxs"])
                        for attr_idx in range(logits.size(2))]
                preds: List[List[List[int]]] = p.starmap(get_pred, args)  # (attr, b, word)
            # attribute loop
            # for attr_idx in range(logits.size(2)):
            #     preds.append([
            #         viterbi(logit)[: len(word_idx) - 1]
            #         for logit, word_idx in zip(logits[:, :, attr_idx, :].detach().cpu(), batch["word_idxs"])
            #     ])

            total_preds.append(preds)
            # test dataの場合truesは使わないので適当にpredsを入れる
            labels = batch["labels"]  # (b, word, attr) or None
            total_trues.append(labels.permute(2, 0, 1).tolist() if labels is not None else preds)

    num_attr: int = len(total_preds[0])
    total_preds_reshaped = [
        [pred for preds in total_preds for pred in preds[attr]]
        for attr in range(num_attr)
    ]  # (attr, N, word)
    total_trues_reshaped = [
        [
            [t for t in true if t != NerDataset.PAD_FOR_LABELS]
            for trues in total_trues for true in trues[attr]
        ]
        for attr in range(num_attr)
    ]  # (attr, N, word)

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


def viterbi(
    logits: torch.Tensor,  # (word, 3)
    penalty=float("inf")
) -> List[int]:
    num_tags = 3

    # 0: O, 1: B, 2: I
    penalties = torch.zeros((num_tags, num_tags))  # (tag, tag)
    penalties[0][2] = penalty
    # penalties[1][1] = penalty  # B -> B も同一属性内ではありえないはず

    pred_tags: List[int] = [0]
    for logit in logits:
        transit_penalty = penalties[pred_tags[-1]]  # (tag)
        tag = (logit - transit_penalty).argmax(dim=-1)  # ()
        pred_tags.append(tag.item())
    return pred_tags[1:]


def parse_arg() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_path", type=str, help="Specify input path in SHINRA2020"
    )
    parser.add_argument(
        "--model_path", type=str, help="Specify path to trained checkpoint"
    )
    parser.add_argument(
        "--shinra_bsz", type=int, default=256, help="Specify number of documents per 1 dataloader"
    )
    parser.add_argument(
        "--mode", type=str, choices=["leaderboard", "all", "final"], default="all",
        help="Specify 'leaderboard' to evaluate leaderboard data and specify 'all' to evaluate all data"
    )
    parser.add_argument(
        "--bert", type=str, default="cl-tohoku/bert-base-japanese", help="Specify attribute_list path in SHINRA2020"
    )

    return parser.parse_args()


def main():
    args = parse_arg()

    bert = AutoModel.from_pretrained(args.bert)
    tokenizer = AutoTokenizer.from_pretrained(args.bert)

    # dataset = [ShinraData(), ....]
    input_path = Path(args.input_path)
    category = input_path.parts[-1]
    shinra_datum = load_shinra_datum(input_path, category, mode=args.mode)

    model = BertForMultilabelNER(bert, len(shinra_datum[0].attributes))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(OrderedDict({k.replace("module.", ""): v for k, v in state_dict.items()}))
    model.to(device)
    model = torch.nn.DataParallel(model)

    save_dir = Path(args.model_path).parent
    with save_dir.joinpath(f"{args.mode}.json").open(mode="wt") as f:
        for tagged_data in [d for d in shinra_datum if d.nes is not None]:
            f.write("\n".join(ne.to_json(ensure_ascii=False) for ne in tagged_data.nes) + "\n")
        for shinra_batch in tqdm(DataLoader(
            [d for d in shinra_datum if d.nes is None],
            batch_size=args.shinra_bsz,
            shuffle=False,
            collate_fn=lambda x: x)):
            for tagged_data in ner_for_shinradata(model, tokenizer, shinra_batch):
                f.write("\n".join(ne.to_json(ensure_ascii=False) for ne in tagged_data.nes) + "\n")


if __name__ == "__main__":
    main()
