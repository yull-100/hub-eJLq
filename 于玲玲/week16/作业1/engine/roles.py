"""角色与阵营定义"""
from enum import Enum


class Role(Enum):
    WEREWOLF = "werewolf"
    SEER = "seer"
    WITCH = "witch"
    VILLAGER = "villager"


class Camp(Enum):
    EVIL = "evil"
    GOOD = "good"


ROLE_CAMP_MAP = {
    Role.WEREWOLF: Camp.EVIL,
    Role.SEER: Camp.GOOD,
    Role.WITCH: Camp.GOOD,
    Role.VILLAGER: Camp.GOOD,
}

ROLE_CN = {
    Role.WEREWOLF: "狼人",
    Role.SEER: "预言家",
    Role.WITCH: "女巫",
    Role.VILLAGER: "村民",
}


class Phase(Enum):
    NIGHT = "night"
    DAY_DISCUSS = "day_discuss"
    DAY_VOTE = "day_vote"
    DAY_SETTLE = "day_settle"
    GAME_OVER = "game_over"
