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
class Dyad:
    id: int
    items: List[Item]

@dataclass
class Result:
    model: str
    triad_id: int
    permutation_id: int
    ranking: Optional[Dict[str, int]]
    response: str

@dataclass
class SDSResult:
    model: str
    item_id: int
    response: str

# Part I structures :::

PART1_ITEMS = [
    Item("FW1","People always have the ability to do otherwise.","FW"),
    Item("FW2","People always have free will.","FW"),
    Item("FW3","How people's lives unfold is completely up to them.","FW"),
    Item("FW4","People ultimately have complete control over their decisions and their actions.","FW"),
    Item("FW5","People have free will even when their choices are completely limited by external circumstances.","FW"),
    Item("DE1","Everything that has ever happened had to happen precisely as it did, given what happened before.","DE"),
    Item("DE2","Every event that has ever occurred, including human decisions and actions, was completely determined by prior events.","DE"),
    Item("DE3","People's choices and actions must happen precisely the way they do because of the laws of nature and the way things were in the distant past.","DE"),
    Item("DE4","A supercomputer that could know everything about the way the universe is now could know everything about the way the universe will be in the future.","DE"),
    Item("DE5","Given the way things were at the Big Bang, there is only one way for everything to happen in the universe after that.","DE"),
    Item("DU1","The fact that we have souls that are distinct from our material bodies is what makes humans unique.","DU"),
    Item("DU2","Each person has a non-physical essence that makes that person unique.","DU"),
    Item("DU3","The human mind cannot simply be reduced to the brain.","DU"),
    Item("DU4","The human mind is more than just a complicated biological machine.","DU"),
    Item("DU5","Human action can only be understood in terms of our souls and minds and not just in terms of our brains.","DU")
]

#Generated from part1_block_design.py
triad_indices = [
    (0, 1, 0),  # FW1, DE2, DU1
    (1, 2, 2),  # FW2, DE3, DU3
    (2, 0, 1),  # FW3, DE1, DU2
    (3, 3, 3),  # FW4, DE4, DU4
    (4, 4, 4),  # FW5, DE5, DU5
    (0, 3, 4),  # FW1, DE4, DU5
    (1, 0, 3),  # FW2, DE1, DU4
    (2, 4, 2),  # FW3, DE5, DU3
    (3, 2, 0),  # FW4, DE3, DU1
    (4, 1, 1),  # FW5, DE2, DU2
]

def build_part1_triads() -> list[Triad]:
    fw = PART1_ITEMS[0:5]
    de = PART1_ITEMS[5:10]
    du = PART1_ITEMS[10:15]

    return [
        Triad(
            id=i,
            items=[fw[a], de[b], du[c]],
        )
        for i, (a, b, c) in enumerate(triad_indices)
    ]

# Part II structures :::

PART2_ITEMS = [
    Item("FC1","Free will is the ability to make different choices even if everything leading up to one's choice were exactly the same.","FC"),
    Item("FC2","Free will is the ability to make a choice based on one's beliefs and desires such that, if one had different beliefs or desires, one's choice would have been different as well.","FC"),
    Item("FC3","People could have free will even if scientists discovered all of the laws that govern all human behavior.","FC"),
    Item("FC4","To have free will means that a person's decisions and actions could not be perfectly predicted by someone else no matter how much information they had.","FC"),
    Item("FC5","If it turned out that people lacked non-physical (or immaterial) souls, then they would lack free will.","FC"),
    Item("FC6","To have free will is to be able to cause things to happen in the world without at the same time being caused to make those things happen.","FC"),
    Item("FC7","People have free will as long as they are able to do what they want without being coerced or constrained by other people.","FC"),
    Item("MC1","To be responsible for our present decisions and actions we must also be responsible for all of our prior decisions and actions that led up to the present moment.","MC"),
    Item("MC2","People deserve to be blamed and punished for bad actions only if they acted of their own free will.","MC"),
    Item("MC3","People who harm others deserve to be punished even if punishing them will not produce any positive benefits to either the offender or society.","MC"),
    Item("MC4","People who perform harmful actions ought to be rehabilitated so they no longer pose a threat to society.","MC"),
    Item("MC5","People who perform harmful actions ought to be punished so that other potential offenders are deterred from committing similar harmful actions.","MC"),
    Item("MC6","People could be morally responsible even if scientists discovered all of the laws that govern human behavior.","MC"),
    Item("MC7","If it turned out that people lacked non-physical (or immaterial) souls, then they would lack moral responsibility.","MC")
]

dyad_indices = [
    (0, 0),  # FC1, MC1
    (1, 1),  # FC2, MC2
    (2, 2),  # FC3, MC3
    (3, 3),  # FC4, MC4
    (4, 6),  # FC5, MC7
    (5, 5),  # FC6, MC6
    (6, 4),  # FC7, MC5
    (0, 5),  # FC1, MC6
    (1, 0),  # FC2, MC1
    (2, 1),  # FC3, MC2
    (3, 4),  # FC4, MC5
    (4, 2),  # FC5, MC3
    (5, 6),  # FC6, MC7
    (6, 3),  # FC7, MC4
]

#Generated from part2_block_design.py
def build_part2_triads() -> list[Dyad]:
    fc = PART2_ITEMS[0:7]
    mc = PART1_ITEMS[7:14]
    
    return [
        Dyad(
            id=i,
            items=[fc[a], mc[b]],
        )
        for i, (a, b) in enumerate(dyad_indices)
    ]
