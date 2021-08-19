import argparse
import os
from pathlib import Path

import joblib
import mlflow
import torch
import torch.nn as nn
import torch.optim as optim
from seqeval.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from dataset import NerDataset, ShinraData, ner_collate_fn
from model import BertForMultilabelNER, create_pooler_matrix
from predict import predict
from util import decode_iob

device = "cuda:0" if torch.cuda.is_available() else "cpu"


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
        "--model_path", type=str, help="Specify attribute_list path in SHINRA2020"
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


def evaluate(model: nn.Module, dataset: Dataset, attributes):
    total_preds, total_trues = predict(model, dataset, device)
    total_preds = decode_iob(total_preds, attributes)
    total_trues = decode_iob(total_trues, attributes)

    f1 = f1_score(total_trues, total_preds)
    return f1


def train(
    model: nn.Module,
    train_dataset: NerDataset,
    valid_dataset: Dataset,
    attributes,
    args,
):
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    # scheduler = get_scheduler(
    #     args.bsz, args.grad_acc, args.epoch, args.warmup, optimizer, len(train_dataset))

    early_stopping = EarlyStopping(patience=10, verbose=1)

    losses = []
    for e in range(args.epoch):
        train_dataloader = DataLoader(
            train_dataset, batch_size=args.bsz, collate_fn=ner_collate_fn, shuffle=True
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
            labels = batch["labels"]  # (attr, b, seq)

            attention_mask = input_ids > 0
            pooling_matrix = create_pooler_matrix(
                input_ids, word_idxs, pool_type="head"
            ).to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels.permute(2, 0, 1),  # (attr, b, seq)
                pooling_matrix=pooling_matrix,
            )

            loss = outputs[0]
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
            torch.save(model.state_dict(), args.model_path + "best.model")

        if e + 1 > 30 and early_stopping.validate(valid_f1):
            break


def main():
    args = parse_arg()

    bert = AutoModel.from_pretrained("cl-tohoku/bert-base-japanese")
    tokenizer = AutoTokenizer.from_pretrained("cl-tohoku/bert-base-japanese")

    category = str(args.input_path).split("/")[-1]

    # dataset = [ShinraData(), ....]
    if os.path.exists(f"../tmp/{category}_dataset.pkl"):
        dataset = joblib.load(f"../tmp/{category}_dataset.pkl")
    else:
        dataset = ShinraData.from_shinra2020_format(Path(args.input_path))
        dataset = [d for d in dataset if d.nes is not None]
        joblib.dump(dataset, f"../tmp/{category}_dataset.pkl", compress=3)

    model = BertForMultilabelNER(bert, len(dataset[0].attributes)).to(device)
    train_dataset, valid_dataset = train_test_split(dataset, test_size=0.1)
    train_dataset = NerDataset(
        [d for train_d in train_dataset for d in train_d.to_ner_examples()], tokenizer
    )
    valid_dataset = NerDataset(
        [d for valid_d in valid_dataset for d in valid_d.to_ner_examples()], tokenizer
    )

    mlflow.start_run()
    mlflow.log_params(vars(args))
    train(model, train_dataset, valid_dataset, dataset[0].attributes, args)
    torch.save(model.state_dict(), args.model_path + "last.model")
    mlflow.end_run()


if __name__ == "__main__":
    main()
