from typing import Iterable, List

import torch
from transformers import get_linear_schedule_with_warmup


def calculate_recall(trues, candidates, ks=[1, 5, 10, 30, 50, 100]):
    results = {}
    for k in ks:
        total = 0
        cnt = 0
        for t, candidate in zip(trues, candidates):
            total += 1
            cnt += int(t in candidate[:k])
        results[k] = cnt / total
    return results


# "O": 0, "B": 1, "I": 2
# 0,1 0,2 1,1 2,1
def is_chunk_start(prev_tag: int, tag: int) -> bool:
    return tag == 1 or (prev_tag == 0 and tag == 2)


# 1,0 2,0 1,1 2,1
def is_chunk_end(prev_tag: int, tag: int) -> bool:
    return prev_tag != 0 and tag != 2


def save_model(model, output_path) -> None:
    model_to_save = model.module if hasattr(model, "module") else model
    torch.save(model_to_save.state_dict(), output_path)


def to_parallel(model):
    model = torch.nn.DataParallel(model)
    return model


def to_fp16(model, optimizer=None, fp16_opt_level=None):
    import apex
    if optimizer is None:
        model = apex.amp.initialize(model, opt_level=fp16_opt_level)
        return model
    else:
        model, optimizer = apex.amp.initialize(
            model, optimizer, opt_level=fp16_opt_level
        )
        return model, optimizer


def get_scheduler(
    batch_size, grad_acc, epochs, warmup_propotion, optimizer, len_train_data
):
    num_train_steps = int(epochs * len_train_data / batch_size / grad_acc)
    num_warmup_steps = int(num_train_steps * warmup_propotion)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps,
        num_train_steps,
    )

    return scheduler


def decode_iob(preds: List[List[Iterable[int]]], attributes: List[str]) -> List[List[str]]:
    iobs: List[List[str]] = []
    idx2iob = ["O", "B", "I"]
    assert len(preds) == len(attributes)
    for attr_iobs, attribute in zip(preds, attributes):
        iobs += [
            [
                idx2iob[idx] + "-" + attribute
                if idx2iob[idx] != "O"
                else "O"
                for idx in iob
            ]
            for iob in attr_iobs
        ]

    return iobs
