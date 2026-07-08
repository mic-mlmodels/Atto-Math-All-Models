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
        input_ids_lst = [entry["input_ids"] for entry in entries]
        attention_mask_lst = [entry["attention_mask"] for entry in entries]
        labels_lst = [entry["labels"] for entry in entries]

        max_length = max([len(tokens) for tokens in input_ids_lst])
        padded_text_tokens_lst = []
        masks_lst = []
        label_lst = []
        for input_ids, attention_mask, labels in zip(
            input_ids_lst, attention_mask_lst, labels_lst
        ):
            padding = max_length - len(input_ids)
            padded_text_tokens_lst.append(
                input_ids + [self.tokeniser.eos_token_id] * padding
            )
            masks_lst.append(attention_mask + [0] * padding)
            label_lst.append(labels + [-100] * padding)
        return {
            "input_ids": torch.tensor(padded_text_tokens_lst),  # type: ignore
            "attention_mask": torch.tensor(masks_lst),
            "labels": torch.tensor(label_lst),
        }
