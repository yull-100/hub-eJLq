"""游戏实例管理器 - 内存存储游戏实例"""
import copy
import threading
from typing import Dict, Optional

from config.settings import GAME_CONFIG
from engine.game_engine import GameEngine


class GameManager:
    """管理多个游戏实例，支持并发访问"""

    def __init__(self):
        self._games: Dict[str, GameEngine] = {}
        self._lock = threading.Lock()
        self._counter = 0

    def create_game(self, config: Optional[dict] = None) -> str:
        """创建新游戏，返回 game_id"""
        with self._lock:
            self._counter += 1
            game_id = f"game_{self._counter}"

            cfg = copy.deepcopy(config or GAME_CONFIG)
            engine = GameEngine(cfg)
            engine.setup()

            self._games[game_id] = engine
            return game_id

    def get_game(self, game_id: str) -> Optional[GameEngine]:
        """获取游戏实例"""
        with self._lock:
            return self._games.get(game_id)

    def step_game(self, game_id: str) -> Optional[dict]:
        """前进一个阶段"""
        engine = self.get_game(game_id)
        if engine is None:
            return None
        with self._lock:
            if engine.state.current_phase.value == "game_over":
                return {"phase": "game_over", "round": engine.state.current_round, "winner": engine.winner}
            return engine.step()

    def run_game(self, game_id: str) -> Optional[dict]:
        """完整运行游戏直到结束"""
        engine = self.get_game(game_id)
        if engine is None:
            return None
        with self._lock:
            while engine.state.current_phase.value != "game_over":
                engine.step()
            return {
                "phase": "game_over",
                "round": engine.state.current_round,
                "winner": engine.winner,
                "log": engine.game_log,
            }

    def get_game_state(self, game_id: str, viewer: Optional[str] = None) -> Optional[dict]:
        """获取游戏状态快照"""
        engine = self.get_game(game_id)
        if engine is None:
            return None
        with self._lock:
            alive_players = [
                {"player_id": p.player_id, "is_alive": p.is_alive}
                for p in engine.state.players
            ]
            return {
                "game_id": game_id,
                "round": engine.state.current_round,
                "phase": engine.state.current_phase.value if hasattr(engine.state.current_phase, "value") else str(engine.state.current_phase),
                "winner": engine.winner,
                "player_count": len(engine.state.players),
                "alive_count": len(engine.state.get_alive_players()),
                "players": alive_players,
                "logs": engine.game_log[-50:],  # 最近 50 条日志
            }

    def get_messages(self, game_id: str, player_id: Optional[str] = None) -> Optional[list]:
        """获取消息历史（支持信息脱敏）"""
        engine = self.get_game(game_id)
        if engine is None:
            return None
        with self._lock:
            if player_id:
                player = engine.state.get_player_by_id(player_id)
                if not player:
                    return []
                msgs = engine.hub.get_visible_messages(player_id, player.camp.value)
            else:
                msgs = engine.hub.messages

            return [
                {
                    "sender": m.sender,
                    "content": m.content,
                    "type": m.msg_type.value,
                    "round": m.round,
                    "phase": m.phase,
                    "target": m.target,
                }
                for m in msgs
            ]

    def list_games(self) -> list:
        """列出所有游戏实例"""
        with self._lock:
            return [
                {
                    "game_id": gid,
                    "round": eng.state.current_round,
                    "phase": eng.state.current_phase.value if hasattr(eng.state.current_phase, "value") else str(eng.state.current_phase),
                    "alive": len(eng.state.get_alive_players()),
                    "total": len(eng.state.players),
                }
                for gid, eng in self._games.items()
            ]

    def delete_game(self, game_id: str) -> bool:
        """删除游戏实例"""
        with self._lock:
            if game_id in self._games:
                del self._games[game_id]
                return True
            return False


# 全局单例
game_manager = GameManager()
