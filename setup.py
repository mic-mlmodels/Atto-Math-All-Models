# %%
# downloads
import os
from datasets import load_dataset
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

cwd = os.getcwd()
print("starting download of the MetaMathQA dataset for the initial sft phase...")
data = load_dataset("meta-math/MetaMathQA", split="train")
data.save_to_disk(os.path.join(cwd, "metamathqa"))
print("finished download :D")

print(
    "starting download of the base model which we are gonna be post-training from scratch!"
)
snapshot_download(
    repo_id="Qwen/Qwen2.5-1.5B", local_dir=os.path.join(cwd, "Qwen2.5-1.5B base model")
)
print("finished download :D")

# %%
# gotta get some tokens first (kind of an ugly solution)
tokeniser = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")
test_chat = [
    {"role": "user", "content": "blah blah"},
    {"role": "assistant", "content": "<think>"},
]
test_text = tokeniser.apply_chat_template(test_chat, tokenize=False)
test_tokens = tokeniser(test_text)["input_ids"]
think_token_id = tokeniser.encode("<think>", add_special_tokens=False)[0]

for i in range(len(test_tokens) - 1, -1, -1):
    if test_tokens[i] == think_token_id:
        end_sequence = test_tokens[i - 3 : i + 1]
        break

print(end_sequence)  # type: ignore

# %%
# processing


def process(entries):
    input_ids = []
    attention_mask = []
    labels = []
    for response, query in zip(entries["response"], entries["query"]):
        if "The answer is:" in response:
            parts = response.rsplit("The answer is:", 1)
            thinking = parts[0].strip()
            answer = parts[1].strip()
            if thinking.endswith(f"#### {answer}"):
                thinking = thinking.rsplit(f"#### {answer}", 1)[0].strip()
            tokens = tokeniser(
                tokeniser.apply_chat_template(
                    [
                        {
                            "role": "system",
                            "content": "You are a helpful assistant. You must think step-by-step inside <think> tags before providing the final answer after ####.",
                        },
                        {"role": "user", "content": query},
                        {
                            "role": "assistant",
                            "content": f"<think>\n{thinking}\n</think>\n#### {answer}",
                        },
                    ],
                    tokenize=False,
                )
            )["input_ids"]
            for i in range(len(tokens)):
                if tokens[i : i + len(end_sequence)] == end_sequence:
                    mask_end_idx = i + len(end_sequence) - 1
                    break
            input_ids.append(tokens)
            attention_mask.append([1] * len(tokens))
            labels.append([-100] * mask_end_idx + tokens[mask_end_idx:])  # type: ignore

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


processed_data = data.map(process, batched=True, remove_columns=data.column_names)
processed_data.save_to_disk(os.path.join(cwd, "processed-metamathqa"))
print("data processed yippee :D")

# %%
# process test
len(processed_data)
processed_data[0]
processed_data[1]

# %%
# process test
data[0]
data[1]
data[2]
data[3]
data[4]

# %%
# process test
