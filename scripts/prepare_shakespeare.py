"""Download tiny-shakespeare and encode it as character-level token arrays."""

import json
import urllib.request
from pathlib import Path

import numpy as np

DATA_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/"
    "tinyshakespeare/input.txt"
)
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "shakespeare"
VAL_FRACTION = 0.1


def download_text(target: Path) -> str:
    """Download the raw corpus once and return its content.

    :param target: File the raw text is cached in.
    :return: The full corpus as a string.
    """
    if not target.exists():
        print(f"downloading {DATA_URL}")
        urllib.request.urlretrieve(DATA_URL, target)
    return target.read_text(encoding="utf-8")


def build_vocab(text: str) -> tuple[dict[str, int], list[str]]:
    """Build the character-level vocabulary of a text.

    :param text: The corpus to derive the vocabulary from.
    :return: Tuple of the character-to-index map and the index-to-character list.
    """
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    return stoi, chars


def main() -> None:
    """Download, encode, split, and store the corpus plus its vocabulary.

    :return: None.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    text = download_text(OUTPUT_DIR / "input.txt")
    stoi, itos = build_vocab(text)

    ids = np.array([stoi[ch] for ch in text], dtype=np.uint16)
    split_at = int(len(ids) * (1 - VAL_FRACTION))
    train_ids, val_ids = ids[:split_at], ids[split_at:]

    train_ids.tofile(OUTPUT_DIR / "train.bin")
    val_ids.tofile(OUTPUT_DIR / "val.bin")
    meta = {"vocab_size": len(itos), "itos": itos}
    (OUTPUT_DIR / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

    print(f"characters: {len(text):,}")
    print(f"vocab size: {len(itos)}")
    print(f"train tokens: {len(train_ids):,}")
    print(f"val tokens:   {len(val_ids):,}")
    print(f"written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
