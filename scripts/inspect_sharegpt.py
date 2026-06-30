from datasets import load_dataset

ds = load_dataset("philschmid/sharegpt-raw", split="train")
print(f"Loaded {len(ds)} conversations")
for i in range(3):
    convs = ds[i]["conversations"]
    print(f"\nSample {i}: {len(convs)} turns")
    for turn in convs[:2]:
        print(f"  [{turn["from"]}: {turn["value"][:120]}...")
