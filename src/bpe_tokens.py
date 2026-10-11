from pathlib import Path

SPECIAL_TEXT = "<|endoftext|>"
SPECIAL_ID = 256
FIRST_MERGE_ID = 257
CACHE_LIMIT = 5_000_000

URDU_WORD_PATTERN = (
    r"""'s|'t|'re|'ve|'m|'ll|'d"""
    r"""| ?[\p{L}\p{M}\u200c]+"""
    r"""| ?\p{N}+"""
    r"""| ?[^\s\p{L}\p{M}\p{N}\u200c]+"""
    r"""|\s+(?!\S)"""
    r"""|\s+"""
)


def open_text_file(path: str | Path):
    return open(path, "r", encoding="utf-8", newline="")


def merge_pair_left_to_right(
    token_ids: list[int],
    pair: tuple[int, int],
    new_id: int,
) -> list[int]:
    left_id, right_id = pair
    merged_ids = []
    position = 0

    while position < len(token_ids):
        is_pair = (
            position + 1 < len(token_ids)
            and token_ids[position] == left_id
            and token_ids[position + 1] == right_id
        )
        if is_pair:
            merged_ids.append(new_id)
            position += 2
        else:
            merged_ids.append(token_ids[position])
            position += 1

    return merged_ids


class PairHeapEntry:
    __slots__ = ("count", "byte_key", "pair")

    def __init__(
        self,
        count: int,
        byte_key: tuple[bytes, bytes],
        pair: tuple[int, int],
    ) -> None:
        self.count = count
        self.byte_key = byte_key
        self.pair = pair

    def __lt__(self, other: "PairHeapEntry") -> bool:
        if self.count != other.count:
            return self.count > other.count
        if self.byte_key != other.byte_key:
            return self.byte_key > other.byte_key
        return self.pair > other.pair
