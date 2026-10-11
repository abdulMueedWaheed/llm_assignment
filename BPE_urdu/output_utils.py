from datetime import datetime
from typing import TextIO


LINE = "=" * 72


class Tee:
    def __init__(self, *streams: TextIO) -> None:
        self.streams = streams

    def write(self, text: str) -> int:
        for stream in self.streams:
            stream.write(text)
        return len(text)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()

    def isatty(self) -> bool:
        return any(stream.isatty() for stream in self.streams)


def show_title(title: str) -> None:
    print(f"\n{LINE}\n{title}\n{LINE}", flush=True)


def show_section(title: str) -> None:
    print(f"\n--- {title} ---", flush=True)


def show_value(name: str, value: object) -> None:
    print(f"{name:<24}: {value}", flush=True)


def clock_time() -> str:
    return datetime.now().astimezone().isoformat(sep=" ", timespec="seconds")


def time_taken(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, rest = divmod(total, 3600)
    minutes, seconds = divmod(rest, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
