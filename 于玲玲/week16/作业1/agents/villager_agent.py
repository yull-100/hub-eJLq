"""村民 Agent"""
import random

from agents.base_agent import BaseAgent
from agents.llm_client import call_llm, call_llm_json, build_game_context
from engine.roles import Role


VILLAGER_SYSTEM_PROMPT = """你是一个狼人杀游戏中的普通村民。
你的核心策略：
1. 你没有特殊技能，但你的投票至关重要
2. 仔细观察每个人的发言逻辑，寻找矛盾和破绽
3. 关注投票记录的异常：谁在保护谁？谁在转移话题？
4. 如果预言家跳身份，配合其引导进行投票
5. 在内心独白中维护一份"嫌疑名单"，根据发言不断更新

记住：你的武器是逻辑和观察力，做好每一次投票。

你必须以 JSON 格式回复决策，不要输出其他内容。"""

VOTE_PROMPT = """现在是投票阶段，你需要投票给一名玩家。
根据发言逻辑分析，投票给你认为最可能是狼人的玩家。
请输出 JSON：{"action": "vote", "target": "Player_X", "reason": "你的理由"}
其中 Player_X 替换为具体的玩家ID，或 "skip" 表示弃票。"""

SPEECH_PROMPT = """现在是白天讨论阶段，你需要公开发言。
你是村民，要通过观察和分析找出狼人。可以表达对某些玩家的怀疑或支持。
请直接输出你的发言内容（纯文本，不要 JSON）。"""

MONOLOGUE_PROMPT = """在发言之前，先进行内心独白分析。
分析：谁的发言有破绽？谁在保护谁？你的嫌疑名单是谁？
请直接输出你的内心分析（纯文本，不要 JSON）。"""


class VillagerAgent(BaseAgent):
    def __init__(self, player_id: str):
        super().__init__(player_id, Role.VILLAGER)

    def _build_system_prompt(self) -> str:
        return VILLAGER_SYSTEM_PROMPT

    def decide_night_action(self, game_info: dict) -> dict:
        return {"action": "none"}

    def _fallback_vote(self, game_info: dict) -> dict:
        alive_others = self.get_alive_others(game_info)
        target = random.choice(alive_others) if alive_others else "skip"
        return {"action": "vote", "target": target, "reason": "根据发言逻辑分析"}

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
        alive_others = self.get_alive_others(game_info)
        if alive_others:
            suspect = random.choice(alive_others)
            return f"我是{self.player_id}，我比较关注{suspect}的发言，觉得有些地方值得推敲。"
        return f"我是{self.player_id}，目前场上信息有限，我还在分析。"

    def generate_speech(self, game_info: dict) -> str:
        if not self.is_llm_enabled():
            return self._fallback_speech(game_info)

        context = build_game_context(game_info, self.memory.inject_to_prompt())
        user_prompt = f"{context}\n\n{SPEECH_PROMPT}"

        result = call_llm(self.system_prompt, user_prompt, temperature=0.8)
        return result if result else self._fallback_speech(game_info)

    def _fallback_monologue(self, game_info: dict) -> str:
        return f"我是村民，没有特殊技能，需要通过发言逻辑来判断谁是狼人。"

    def inner_monologue(self, game_info: dict) -> str:
        if not self.is_llm_enabled():
            return self._fallback_monologue(game_info)

        context = build_game_context(game_info, self.memory.inject_to_prompt())
        user_prompt = f"{context}\n\n{MONOLOGUE_PROMPT}"

        result = call_llm(self.system_prompt, user_prompt, temperature=0.7)
        return result if result else self._fallback_monologue(game_info)
