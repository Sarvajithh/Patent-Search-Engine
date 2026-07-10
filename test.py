# from datasets import load_dataset

# ds = load_dataset("ag_news", split="train", streaming=True)
# print(next(iter(ds)))

from datasets import get_dataset_config_names

print(get_dataset_config_names("big_patent"))