import glob
from pathlib import Path
import sentencepiece as spm
import os

DATA_CACHE_DIR = Path("data")
# data_dir = DATA_CACHE_DIR / "TinyStories_all_data"
# shard_filenames = sorted(glob.glob(str(data_dir / "*.json")))

# print(shard_filenames)

from datasets import load_dataset

dataset = load_dataset("factored/us_patent_hub")

descriptions = dataset["train"]["description"][:20000]

joined_text = "\n".join(descriptions)

with open("data/patent_desc.txt", "w", encoding="utf-8") as file:
    file.write(joined_text)

prefix = DATA_CACHE_DIR / "tokPatent"
file = DATA_CACHE_DIR / "patent_desc.txt"

spm.SentencePieceTrainer.train(
        input=str(file),
        model_prefix=str(prefix),
        model_type="bpe",
        vocab_size=4096,
        self_test_sample_size=0,
        input_format="text",
        num_threads=os.cpu_count(),
        split_digits=True,
        allow_whitespace_only_pieces=True,
        byte_fallback=True,
        unk_surface=r"\342\201\207 ",
        normalization_rule_name="identity",
    )