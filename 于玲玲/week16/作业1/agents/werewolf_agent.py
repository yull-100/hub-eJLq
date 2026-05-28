"""狼人 Agent"""
import random

from agents.base_agent import BaseAgent
from agents.llm_client import call_llm, call_llm_json, build_game_context
from engine.roles import Role


WEREWOLF_SYSTEM_PROMPT = """你是一个狼人杀游戏中的狼人。
你的核心策略：
1. 白天伪装成好人，不要暴露狼人身份
2. 发言时制造混乱，误导好人阵营的判断
3. 与队友配合，在投票中保护彼此
4. 夜间优先击杀有威胁的角色（如预言家、女巫）
5. 在内心独白中分析场上局势，制定击杀策略

记住：你的目标是消灭所有好人，同时隐藏自己的身份。

你必须以 JSON 格式回复，不要输出其他内容。"""

NIGHT_PROMPT = """现在是夜晚，你需要选择击杀目标。
存活的好人玩家是潜在的击杀对象。
请输出 JSON：{"action": "kill", "target": "Player_X"}
其中 Player_X 替换为具体的玩家ID。"""

VOTE_PROMPT = """现在是投票阶段，你需要投票给一名玩家。
作为狼人，你要伪装成好人，投票给可疑的玩家，但不要投给队友。
请输出 JSON：{"action": "vote", "target": "Player_X", "reason": "你的理由"}
其中 Player_X 替换为具体的玩家ID，或 "skip" 表示弃票。"""

SPEECH_PROMPT = """现在是白天讨论阶段，你需要公开发言。
作为狼人，你要伪装成好人，发言要自然，可以分析局势但不要暴露身份。
请直接输出你的发言内容（纯文本，不要 JSON）。"""

MONOLOGUE_PROMPT = """在发言之前，先进行内心独白分析。
分析当前局势：谁是威胁？队友是否安全？今晚应该击杀谁？
请直接输出你的内心分析（纯文本，不要 JSON）。"""


class WerewolfAgent(BaseAgent):
    def __init__(self, player_id: str):
        super().__init__(player_id, Role.WEREWOLF)

    def _build_system_prompt(self) -> str:
        return WEREWOLF_SYSTEM_PROMPT

    def _fallback_night_action(self, game_info: dict) -> dict:
        """随机策略回退"""
        alive_others = self.get_alive_others(game_info)
        known_roles = game_info.get("known_roles", {})
        targets = [p for p in alive_others if known_roles.get(p) != "werewolf"]
        target = random.choice(targets) if targets else alive_others[0]
        return {"action": "kill", "target": target}

    def decide_night_action(self, game_info: dict) -> dict:
        if not self.is_llm_enabled():
            return self._fallback_night_action(game_info)

        context = build_game_context(game_info, self.memory.inject_to_prompt())
        user_prompt = f"{context}\n\n{NIGHT_PROMPT}"

        result = call_llm_json(self.system_prompt, user_prompt, temperature=0.5)
        if result and "target" in result:
            return {"action": "kill", "target": result["target"]}
        return self._fallback_night_action(game_info)

    def _fallback_vote(self, game_info: dict) -> dict:
        alive_others = self.get_alive_others(game_info)
        target = random.choice(alive_others) if alive_others else "skip"
        return {"action": "vote", "target": target, "reason": "根据发言逻辑判断"}

    def decide_vote(self, game_info: dict) -> dict:
        if not self.is_llm_enabled():
            return self._fallback_vote(game_info)

        context = build_game_context(game_info, self.memory.inject_to_prompt())
        user_prompt = f"{context}\n\n{VOTE_PROMPT}"

        result = call_llm_json(self.system_prompt, user_prompt, temperature=0.7)
        if result and "target" in result:
            return {
                "action": "vote",
                "target": result["target"],
                "reason": result.get("reason", "综合分析判断"),
            }
        return self._fallback_vote(game_info)

    def _fallback_speech(self, game_info: dict) -> str:
        return f"我是{self.player_id}，我觉得我们应该仔细分析每个人的发言逻辑，找出其中的破绽。"

    def generate_speech(self, game_info: dict) -> str:
        if not self.is_llm_enabled():
            return self._fallback_speech(game_info)

        context = build_game_context(game_info, self.memory.inject_to_prompt())
        user_prompt = f"{context}\n\n{SPEECH_PROMPT}"

        result = call_llm(self.system_prompt, user_prompt, temperature=0.8)
        return result if result else self._fallback_speech(game_info)

    def _fallback_monologue(self, game_info: dict) -> str:
        return f"我是狼人，队友是{self.teammates}。需要隐藏身份，找机会击杀关键角色。"

    def inner_monologue(self, game_info: dict) -> str:
        if not self.is_llm_enabled():
            return self._fallback_monologue(game_info)

        context = build_game_context(game_info, self.memory.inject_to_prompt())
        user_prompt = f"{context}\n\n队友是 {self.teammates}。\n\n{MONOLOGUE_PROMPT}"

        result = call_llm(self.system_prompt, user_prompt, temperature=0.7)
        return result if result else self._fallback_monologue(game_info)
