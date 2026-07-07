# %%
# downloads
import inspect
import os
from datasets import load_dataset
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, InputFeatures

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
# processing
tokeniser = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")


def process(entries):
    input_ids = []
    attention_mask = []
    labels = []
    for response, query in zip(entries["response"], entries["query"]):
        if "The answer is:" in response:
            parts = response.rsplit("The answer is:", 1)
            thinking = parts[0].strip()
            answer = parts[1].strip()
            full_lst = inspect.cleandoc(f"""
                <|im_start|>system
                You are a helpful assistant. You must think step-by-step inside <think> tags before providing the final answer after ####.<|im_end|>
                <|im_start|>user
                {query}<|im_end|>
                <|im_start|>assistant
                <think>
                {thinking}
                </think>
                #### {answer}<|im_end|>
            """)

            thinking_lst = inspect.cleandoc(f"""
                <think>
                {thinking}
                </think>
                #### {answer}<|im_end|>
            """)
            full_lst = tokeniser(full_lst)
            thinking_lst = tokeniser(thinking_lst)

            input_ids.append(full_lst["input_ids"])
            attention_mask.append(full_lst["attention_mask"])
            labels.append(
                [-100] * (len(full_lst["input_ids"]) - len(thinking_lst["input_ids"]))
                + thinking_lst["input_ids"]
            )
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


processed_data = data.map(process, batched=True, remove_columns=data.column_names)
processed_data.save_to_disk(os.path.join(cwd, "processed metamathqa"))
print("data processed yippee :D")

# %%
# process test
len(processed_data)
processed_data[0]
processed_data[1]
