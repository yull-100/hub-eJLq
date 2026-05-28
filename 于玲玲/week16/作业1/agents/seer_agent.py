"""预言家 Agent"""
import random

from agents.base_agent import BaseAgent
from agents.llm_client import call_llm, call_llm_json, build_game_context
from engine.roles import Role


SEER_SYSTEM_PROMPT = """你是一个狼人杀游戏中的预言家。
你的核心策略：
1. 每晚查验一名可疑玩家的身份
2. 白天适时跳身份，引导好人阵营投票
3. 注意保护自己，避免过早暴露被狼人击杀
4. 在内心独白中分析场上发言，找出最可疑的玩家进行查验

记住：你是好人阵营最重要的信息来源，要平衡信息共享与自身安全。

你必须以 JSON 格式回复决策，不要输出其他内容。"""

NIGHT_PROMPT = """现在是夜晚，你需要查验一名玩家的身份。
选择一个你认为最可疑的玩家进行查验。
请输出 JSON：{"action": "check", "target": "Player_X"}
其中 Player_X 替换为具体的玩家ID。"""

VOTE_PROMPT = """现在是投票阶段，你需要投票给一名玩家。
根据你的查验结果和发言分析，投票给你认为最可能是狼人的玩家。
请输出 JSON：{"action": "vote", "target": "Player_X", "reason": "你的理由"}
其中 Player_X 替换为具体的玩家ID，或 "skip" 表示弃票。"""

SPEECH_PROMPT = """现在是白天讨论阶段，你需要公开发言。
你是预言家，可以考虑是否跳身份。如果跳身份，要引导大家投票狼人；如果不跳，要给出有价值的分析。
请直接输出你的发言内容（纯文本，不要 JSON）。"""

MONOLOGUE_PROMPT = """在发言之前，先进行内心独白分析。
分析：谁最可疑？是否应该跳身份？如何保护自己和引导好人阵营？
请直接输出你的内心分析（纯文本，不要 JSON）。"""


class SeerAgent(BaseAgent):
    def __init__(self, player_id: str):
        super().__init__(player_id, Role.SEER)

    def _build_system_prompt(self) -> str:
        return SEER_SYSTEM_PROMPT

    def _fallback_night_action(self, game_info: dict) -> dict:
        alive_others = self.get_alive_others(game_info)
        target = random.choice(alive_others) if alive_others else self.player_id
        return {"action": "check", "target": target}

    def decide_night_action(self, game_info: dict) -> dict:
        if not self.is_llm_enabled():
            return self._fallback_night_action(game_info)

        context = build_game_context(game_info, self.memory.inject_to_prompt())
        user_prompt = f"{context}\n\n{NIGHT_PROMPT}"

        result = call_llm_json(self.system_prompt, user_prompt, temperature=0.5)
        if result and "target" in result:
            return {"action": "check", "target": result["target"]}
        return self._fallback_night_action(game_info)

    def _fallback_vote(self, game_info: dict) -> dict:
        alive_others = self.get_alive_others(game_info)
        target = random.choice(alive_others) if alive_others else "skip"
        return {"action": "vote", "target": target, "reason": "综合分析判断"}

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
        return f"我是{self.player_id}，我有一些关于局势的判断想和大家分享，但目前还在观察阶段。"

    def generate_speech(self, game_info: dict) -> str:
        if not self.is_llm_enabled():
            return self._fallback_speech(game_info)

        context = build_game_context(game_info, self.memory.inject_to_prompt())
        user_prompt = f"{context}\n\n{SPEECH_PROMPT}"

        result = call_llm(self.system_prompt, user_prompt, temperature=0.8)
        return result if result else self._fallback_speech(game_info)

    def _fallback_monologue(self, game_info: dict) -> str:
        return f"我是预言家，需要谨慎选择查验目标，同时考虑是否要暴露身份引导投票。"

    def inner_monologue(self, game_info: dict) -> str:
        if not self.is_llm_enabled():
            return self._fallback_monologue(game_info)

        context = build_game_context(game_info, self.memory.inject_to_prompt())
        user_prompt = f"{context}\n\n{MONOLOGUE_PROMPT}"

        result = call_llm(self.system_prompt, user_prompt, temperature=0.7)
        return result if result else self._fallback_monologue(game_info)
