import json
from pathlib import Path
from typing import Dict, List, Tuple


class BPETokenizer:
    """Byte-Level Byte-Pair Encoding (BPE) subword tokenizer built from scratch."""

    BOS_ID = 256
    EOS_ID = 257
    PAD_ID = 258
    BASE_VOCAB = 259  # 256 raw UTF-8 bytes + 3 special tokens

    def __init__(self) -> None:
        self.merges: Dict[Tuple[int, int], int] = {}
        self.vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        self.vocab[self.BOS_ID] = b"<|bos|>"
        self.vocab[self.EOS_ID] = b"<|eos|>"
        self.vocab[self.PAD_ID] = b"<|pad|>"

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def train(self, text: str, target_vocab_size: int) -> None:
        if target_vocab_size < self.BASE_VOCAB:
            raise ValueError(f"target_vocab_size must be >= {self.BASE_VOCAB}")

        ids = list(text.encode("utf-8"))
        num_merges = target_vocab_size - self.BASE_VOCAB

        for i in range(num_merges):
            pair_counts = self._count_pairs(ids)
            if not pair_counts:
                break
            best_pair = max(pair_counts, key=lambda p: (pair_counts[p], -p[0], -p[1]))
            if pair_counts[best_pair] < 2:
                break
            new_id = self.BASE_VOCAB + i
            ids = self._merge_pair(ids, best_pair, new_id)
            self.merges[best_pair] = new_id
            self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        ids = list(text.encode("utf-8"))
        while len(ids) >= 2:
            pair_counts = self._count_pairs(ids)
            # Pick the applicable merge pair with the lowest merge rank (learned earliest)
            best_pair = min(
                pair_counts.keys(),
                key=lambda p: self.merges.get(p, float("inf")),
            )
            if best_pair not in self.merges:
                break
            ids = self._merge_pair(ids, best_pair, self.merges[best_pair])

        if add_bos:
            ids.insert(0, self.BOS_ID)
        if add_eos:
            ids.append(self.EOS_ID)
        return ids

    def decode(self, ids: List[int]) -> str:
        raw = b"".join(
            self.vocab[idx]
            for idx in ids
            if idx not in (self.BOS_ID, self.EOS_ID, self.PAD_ID) and idx in self.vocab
        )
        return raw.decode("utf-8", errors="replace")

    def save(self, path: str | Path) -> None:
        serialized = {
            "merges": [[p[0], p[1], idx] for p, idx in self.merges.items()],
        }
        Path(path).write_text(json.dumps(serialized, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        tok = cls()
        for p0, p1, idx in data["merges"]:
            pair = (int(p0), int(p1))
            idx = int(idx)
            tok.merges[pair] = idx
            tok.vocab[idx] = tok.vocab[pair[0]] + tok.vocab[pair[1]]
        return tok

    @staticmethod
    def _count_pairs(ids: List[int]) -> Dict[Tuple[int, int], int]:
        counts: Dict[Tuple[int, int], int] = {}
        for a, b in zip(ids, ids[1:]):
            pair = (a, b)
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    @staticmethod
    def _merge_pair(ids: List[int], pair: Tuple[int, int], new_id: int) -> List[int]:
        out: List[int] = []
        i = 0
        n = len(ids)
        while i < n:
            if i + 1 < n and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                out.append(new_id)
                i += 2
            else:
                out.append(ids[i])
                i += 1
        return out
