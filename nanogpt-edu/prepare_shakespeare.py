"""Download TinyShakespeare and write character-level train/val token shards."""
import os, pickle, urllib.request
import numpy as np

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA = os.path.join(os.path.dirname(__file__), "data")


def main():
    os.makedirs(DATA, exist_ok=True)
    raw = os.path.join(DATA, "input.txt")
    if not os.path.exists(raw):
        print(f"downloading {URL}")
        urllib.request.urlretrieve(URL, raw)
    text = open(raw, encoding="utf-8").read()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    print(f"vocab size: {len(chars)} chars, {len(text):,} total chars")

    n = int(0.9 * len(text))
    train_ids = np.array([stoi[c] for c in text[:n]], dtype=np.uint16)
    val_ids = np.array([stoi[c] for c in text[n:]], dtype=np.uint16)
    train_ids.tofile(os.path.join(DATA, "train.bin"))
    val_ids.tofile(os.path.join(DATA, "val.bin"))
    with open(os.path.join(DATA, "meta.pkl"), "wb") as f:
        pickle.dump({"vocab_size": len(chars), "stoi": stoi, "itos": itos}, f)
    print(f"train={len(train_ids):,} val={len(val_ids):,}")


if __name__ == "__main__":
    main()
