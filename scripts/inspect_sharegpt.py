from datasets import load_dataset

ds = load_dataset("philschmid/sharegpt-raw", split="train")
print(f"Loaded {len(ds)} conversations")

for i in range(3):
    print(f"
Sample {i}: {len(ds[i]['conversations'])} turns")
    for turn in ds[i]["conversations"][:2]:
        print(f"  [{turn['from']}]: {turn['value'][:120]}...")
