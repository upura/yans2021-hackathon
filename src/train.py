import argparse
import os
from pathlib import Path
from typing import List
from datetime import datetime

import joblib
import mlflow
import torch
import torch.nn as nn
import torch.optim as optim
from seqeval.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from dataset.shinra import ShinraData
from dataset.ner import NerDataset, ner_collate_fn
from dataset.pseudo import PseudoDataset
from model import BertForMultilabelNER, create_pooler_matrix
from predict import predict
from util import decode_iob

# device = "cuda:0" if torch.cuda.is_available() else "cpu"


class EarlyStopping:
    def __init__(self, patience=0, verbose=0):
        self._step = 0
        self._score = -float("inf")
        self.patience = patience
        self.verbose = verbose

    def validate(self, score):
        if self._score > score:
            self._step += 1
            if self._step > self.patience:
                if self.verbose:
                    print("early stopping")
                return True
        else:
            self._step = 0
            self._score = score

        return False


def parse_arg() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_path", type=str, help="Specify input path in SHINRA2020"
    )
    parser.add_argument(
        "--pseudo_input_path", "-p", type=str, help="Specify input path in SHINRA2019"
    )
    parser.add_argument(
        "--save_path", type=str, help="Specify path to directory where trained checkpoints are saved"
    )
    parser.add_argument(
        "--additional_name", "-a", type=str, help="Specify any string to identify experiment condition"
    )
    parser.add_argument(
        "--lr", type=float, help="Specify attribute_list path in SHINRA2020"
    )
    parser.add_argument(
        "--bsz", type=int, help="Specify attribute_list path in SHINRA2020"
    )
    parser.add_argument(
        "--epoch", type=int, help="Specify attribute_list path in SHINRA2020"
    )
    parser.add_argument(
        "--grad_acc", type=int, help="Specify attribute_list path in SHINRA2020"
    )
    parser.add_argument(
        "--grad_clip", type=float, help="Specify attribute_list path in SHINRA2020"
    )
    parser.add_argument(
        "--note", type=str, help="Specify attribute_list path in SHINRA2020"
    )

    return parser.parse_args()


def evaluate(model: nn.Module, dataset: NerDataset, attributes: List[str]):
    total_preds, total_trues = predict(model, dataset, sent_wise=False)
    total_preds = decode_iob(total_preds, attributes)
    total_trues = decode_iob(total_trues, attributes)

    f1 = f1_score(total_trues, total_preds)
    return f1


def train(
    model: nn.Module,
    train_dataset: NerDataset,
    valid_dataset: NerDataset,
    attributes: List[str],
    save_dir: Path,
    args: argparse.Namespace,
):
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    # scheduler = get_scheduler(
    #     args.bsz, args.grad_acc, args.epoch, args.warmup, optimizer, len(train_dataset))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    early_stopping = EarlyStopping(patience=10, verbose=1)

    # setup model
    model.to(device)
    model = torch.nn.DataParallel(model)

    losses = []
    for e in range(args.epoch):
        train_dataloader = DataLoader(
            train_dataset, batch_size=args.bsz, collate_fn=ner_collate_fn, shuffle=True, num_workers=4,
        )
        bar = tqdm(total=len(train_dataset))

        total_loss = 0
        model.train()
        for step, batch in enumerate(train_dataloader):
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            input_ids = batch["input_ids"]  # (b, seq)
            word_idxs = batch["word_idxs"]  # (b, word)
            labels = batch["labels"]  # (b, word, attr)

            attention_mask = input_ids > 0
            pooling_matrix = create_pooler_matrix(
                input_ids, word_idxs, pool_type="head"
            ).to(device)

            loss, output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                pooling_matrix=pooling_matrix,
            )

            if len(loss.size()) > 0:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training

            loss.backward()

            total_loss += loss.item() * input_ids.size(0)
            mlflow.log_metric("Train batch loss", loss.item(), step=(e + 1) * step)

            bar.set_description(f"[Epoch] {e + 1}")
            bar.set_postfix({"loss": loss.item()})
            bar.update(args.bsz)

            if (step + 1) % args.grad_acc == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                # scheduler.step()
                optimizer.zero_grad()

        losses.append(total_loss / len(train_dataset))
        mlflow.log_metric("Train loss", losses[-1], step=e)

        valid_f1 = evaluate(model, valid_dataset, attributes)
        mlflow.log_metric("Valid F1", valid_f1, step=e)

        if early_stopping._score < valid_f1:
            torch.save(model.state_dict(), save_dir / "best.model")

        if e + 1 > 30 and early_stopping.validate(valid_f1):
            break


def main():
    args = parse_arg()

    bert = AutoModel.from_pretrained("cl-tohoku/bert-base-japanese")
    tokenizer = AutoTokenizer.from_pretrained("cl-tohoku/bert-base-japanese")

    # dataset = [ShinraData(), ....]
    category = Path(args.input_path).parts[-1]
    dataset_cache_dir = Path(os.environ.get("SHINRA_CACHE_DIR", "../tmp"))
    dataset_cache_dir.mkdir(exist_ok=True)
    cache_path = dataset_cache_dir / f"{category}_train_dataset.pkl"
    if cache_path.exists():
        print(f"Loading cached dataset from {cache_path}")
        shinra_datum = joblib.load(cache_path)
    else:
        print(f"Cached shinra_datum not found. Building one from {args.input_path}")
        shinra_datum = ShinraData.from_shinra2020_format(Path(args.input_path), mode="train")
        joblib.dump(shinra_datum, cache_path, compress=3)

    model = BertForMultilabelNER(bert, len(shinra_datum[0].attributes))

    train_shinra_datum, valid_shinra_datum = train_test_split(shinra_datum, test_size=0.1)
    train_dataset = NerDataset.from_shinra(train_shinra_datum, tokenizer)
    valid_dataset = NerDataset.from_shinra(valid_shinra_datum, tokenizer)

    if args.pseudo_input_path:
        pseudo_shinra_datum = ShinraData.from_shinra2020_format(Path(args.input_path), mode="pseudo")
        pseudo_train_dataset = PseudoDataset.from_shinra(pseudo_shinra_datum, args.pseudo_input_path, tokenizer)
        train_dataset = ConcatDataset((train_dataset, pseudo_train_dataset))

    save_dir = Path(args.save_path).joinpath(
        f"{category}{datetime.now().strftime(r'%m%d_%H%M')}" +
        (f"_{args.additional_name}" if args.additional_name else "")
    )
    save_dir.mkdir(parents=True)

    mlflow.start_run()
    mlflow.log_params(vars(args))
    train(model, train_dataset, valid_dataset, shinra_datum[0].attributes, save_dir, args)
    torch.save(model.state_dict(), save_dir / "last.model")
    mlflow.end_run()


if __name__ == "__main__":
    main()
