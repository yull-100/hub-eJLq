"""全局游戏状态管理"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

from engine.roles import Role, Camp, Phase, ROLE_CAMP_MAP


@dataclass
class Player:
    player_id: str
    role: Role
    camp: Camp
    is_alive: bool = True


class GameState:
    def __init__(self):
        self.players: List[Player] = []
        self.current_round: int = 0
        self.current_phase: Phase = Phase.NIGHT
        self.vote_records: Dict[int, Dict[str, str]] = {}
        self.night_actions: Dict[int, Dict[str, Any]] = {}
        self.deaths: List[dict] = []
        self.witch_antidote_used: bool = False
        self.witch_poison_used: bool = False

    def add_player(self, player: Player):
        self.players.append(player)

    def get_alive_players(self) -> List[Player]:
        return [p for p in self.players if p.is_alive]

    def get_alive_players_by_role(self, role: Role) -> List[Player]:
        return [p for p in self.players if p.is_alive and p.role == role]

    def get_alive_players_by_camp(self, camp: Camp) -> List[Player]:
        return [p for p in self.players if p.is_alive and p.camp == camp]

    def get_player_by_id(self, player_id: str) -> Optional[Player]:
        for p in self.players:
            if p.player_id == player_id:
                return p
        return None

    def kill_player(self, player_id: str, cause: str = "unknown"):
        player = self.get_player_by_id(player_id)
        if player and player.is_alive:
            player.is_alive = False
            self.deaths.append({
                "round": self.current_round,
                "player_id": player_id,
                "cause": cause,
            })

    def check_win_condition(self) -> Optional[str]:
        """检查胜负条件，返回获胜阵营或 None"""
        alive_werewolves = self.get_alive_players_by_camp(Camp.EVIL)
        alive_good = self.get_alive_players_by_camp(Camp.GOOD)

        if len(alive_werewolves) == 0:
            return "good"
        if len(alive_werewolves) >= len(alive_good):
            return "evil"
        return None

    def get_snapshot_for_player(self, viewer_id: str) -> dict:
        """为指定玩家生成脱敏后的状态快照"""
        viewer = self.get_player_by_id(viewer_id)
        if not viewer:
            return {}

        alive_players = [
            {"player_id": p.player_id, "is_alive": p.is_alive}
            for p in self.players
        ]

        # 狼人可以看到其他狼人的身份
        known_roles = {}
        if viewer.camp == Camp.EVIL:
            for p in self.players:
                if p.camp == Camp.EVIL:
                    known_roles[p.player_id] = p.role.value
        # 所有人都知道自己的身份
        known_roles[viewer_id] = viewer.role.value

        return {
            "round": self.current_round,
            "phase": self.current_phase.value,
            "my_id": viewer_id,
            "my_role": viewer.role.value,
            "my_camp": viewer.camp.value,
            "alive_players": alive_players,
            "known_roles": known_roles,
            "deaths": [
                {"player_id": d["player_id"], "round": d["round"]}
                for d in self.deaths
            ],
            "vote_records": self.vote_records,
        }
