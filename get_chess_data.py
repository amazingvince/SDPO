import os

from datasets import load_dataset

ds = load_dataset("amazingvince/chess-traces")

if "train" in ds and "test" in ds:
    train_ds = ds["train"]
    test_ds = ds["test"]
else:
    split = ds["train"].train_test_split(test_size=0.001, seed=42, shuffle=True)
    train_ds = split["train"]
    test_ds = split["test"]

# Limit dataset sizes
TRAIN_SIZE = 100_000
TEST_SIZE = 100

train_ds = train_ds.select(range(min(TRAIN_SIZE, len(train_ds))))
test_ds = test_ds.select(range(min(TEST_SIZE, len(test_ds))))

os.makedirs("datasets/chess", exist_ok=True)
train_ds.to_json("datasets/chess/train.jsonl")
test_ds.to_json("datasets/chess/test.jsonl")
