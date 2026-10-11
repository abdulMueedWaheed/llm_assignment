"""Helpers for writing command output to both the terminal and a report file."""

from datetime import datetime
from typing import TextIO


SEPARATOR = "=" * 76


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


def print_header(title: str) -> None:
    print(f"\n{SEPARATOR}\n{title}\n{SEPARATOR}", flush=True)


def print_section(title: str) -> None:
    print(f"\n--- {title} ---", flush=True)


def print_field(label: str, value: object) -> None:
    print(f"{label:<24} : {value}", flush=True)


def local_timestamp() -> str:
    return datetime.now().astimezone().isoformat(sep=" ", timespec="seconds")


def format_duration(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"
