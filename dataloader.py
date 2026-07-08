import random
import torch


class Dataloader:
    def __init__(self, dataset, shuffle, tokeniser, batch_size):
        self.dataset = dataset
        self.shuffle = shuffle
        self.tokeniser = tokeniser
        self.batch_size = batch_size

    def __iter__(self):
        iter_lst = list(range(len(self.dataset)))
        if self.shuffle:
            random.shuffle(iter_lst)
        for entry_idx in range(0, len(iter_lst), self.batch_size):
            yield self.extract(
                [
                    self.dataset[i]
                    for i in iter_lst[entry_idx : entry_idx + self.batch_size]
                ]
            )

    def extract(self, entries):
        padded_text_tokens_lst = []
        masks_lst = []
        label_lst = []
        new_padded_text_tokens_lst = []
        new_masks_lst = []
        new_label_lst = []
        for entry in entries:
            padded_text_tokens_lst.append(entry["input_ids"])
            masks_lst.append(entry["attention_mask"])
            label_lst.append(entry["labels"])
        max_length = max([len(tokens) for tokens in padded_text_tokens_lst])
        for padded_text_tokens in padded_text_tokens_lst:
            padding = max_length - len(padded_text_tokens)
            new_padded_text_tokens_lst.append(
                padded_text_tokens + [self.tokeniser.eos_token_id] * padding
            )
        for padded_text_tokens in padded_text_tokens_lst:
            padding = max_length - len(padded_text_tokens)
            new_padded_text_tokens_lst.append(
                padded_text_tokens + [self.tokeniser.eos_token_id] * padding
            )

        for padded_text_tokens in padded_text_tokens_lst:
            padding = max_length - len(padded_text_tokens)
            new_padded_text_tokens_lst.append(
                padded_text_tokens + [self.tokeniser.eos_token_id] * padding
            )

        return {
            "input_ids": torch.tensor(padded_text_tokens_lst),  # type: ignore
            "attention_mask": torch.tensor(masks_lst),
            "labels": torch.tensor(label_lst),
        }
