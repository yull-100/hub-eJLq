"""Agent 抽象基类"""
from abc import ABC, abstractmethod
from typing import List, Optional

from engine.roles import Role, Camp
from agents.memory import Memory
from config.settings import LLM_CONFIG


class BaseAgent(ABC):
    def __init__(self, player_id: str, role: Optional[Role] = None):
        self.player_id = player_id
        self.role = role
        self.camp: Optional[Camp] = None
        self.teammates: List[str] = []
        self.memory = Memory(player_id)
        self.system_prompt = self._build_system_prompt()

    def set_role_info(self, role: Role, camp: Camp):
        self.role = role
        self.camp = camp
        self.system_prompt = self._build_system_prompt()

    def set_teammates(self, teammates: List[str]):
        self.teammates = teammates

    def is_llm_enabled(self) -> bool:
        """检查 LLM 是否启用"""
        return LLM_CONFIG.get("enabled", False)

    @abstractmethod
    def _build_system_prompt(self) -> str:
        """构建角色专属的 System Prompt"""
        pass

    @abstractmethod
    def decide_night_action(self, game_info: dict) -> dict:
        """夜间决策，返回结构化动作 dict"""
        pass

    @abstractmethod
    def decide_vote(self, game_info: dict) -> dict:
        """投票决策，返回 {"action": "vote", "target": "...", "reason": "..."}"""
        pass

    @abstractmethod
    def generate_speech(self, game_info: dict) -> str:
        """白天公开发言"""
        pass

    @abstractmethod
    def inner_monologue(self, game_info: dict) -> str:
        """内心独白（ToM），不对外广播"""
        pass

    def receive_message(self, message: dict):
        """接收消息并更新记忆"""
        if "speaker" in message and "content" in message:
            self.memory.add_conversation(
                message.get("round", 0),
                message["speaker"],
                message["content"],
            )

    def update_memory(self, event: dict):
        """更新记忆"""
        event_type = event.get("type", "unknown")
        if event_type == "vote":
            self.memory.add_vote_record(event["round"], event["votes"])
        elif event_type == "death":
            self.memory.add_event(event["round"], "death", event.get("detail", ""))
        elif event_type == "claim":
            self.memory.add_identity_claim(event["player_id"], event["role"])

    def get_alive_others(self, game_info: dict) -> List[str]:
        """获取存活的其他玩家 ID"""
        return [
            p["player_id"] for p in game_info.get("alive_players", [])
            if p["is_alive"] and p["player_id"] != self.player_id
        ]
