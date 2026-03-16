from dataclasses import dataclass
from typing import List, Dict, Optional


@dataclass
class Item:
    id: str
    text: str
    dimension: str


@dataclass
class Triad:
    id: int
    items: List[Item]


@dataclass
class Result:
    model: str
    triad_id: int
    permutation_id: int
    ranking: Optional[Dict[str, int]]
    prompt: str