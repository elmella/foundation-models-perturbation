from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import TypeVar


T = TypeVar("T")


def progress(
    iterable: Iterable[T],
    *,
    desc: str | None = None,
    total: int | None = None,
    unit: str = "it",
) -> Iterator[T]:
    try:
        from tqdm.auto import tqdm
    except ImportError:
        yield from iterable
        return
    yield from tqdm(iterable, desc=desc, total=total, unit=unit)


def progress_range(
    *args: int,
    desc: str | None = None,
    unit: str = "it",
) -> Iterator[int]:
    values = range(*args)
    yield from progress(values, desc=desc, total=len(values), unit=unit)

