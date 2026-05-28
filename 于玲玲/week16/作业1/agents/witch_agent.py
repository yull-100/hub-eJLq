"""女巫 Agent"""
import random

from agents.base_agent import BaseAgent
from agents.llm_client import call_llm, call_llm_json, build_game_context
from engine.roles import Role


WITCH_SYSTEM_PROMPT = """你是一个狼人杀游戏中的女巫。
你的核心策略：
1. 你有一瓶解药和一瓶毒药，各只能使用一次
2. 解药可以救活当晚被狼人击杀的人，毒药可以毒杀任意一人
3. 解药优先救关键角色（如预言家），毒药用于确定性极高的狼人
4. 白天不要过早暴露女巫身份，保持信息优势
5. 在内心独白中分析谁值得救、谁应该被毒

记住：你是好人阵营的最后一道防线，用药时机至关重要。

你必须以 JSON 格式回复决策，不要输出其他内容。"""

NIGHT_PROMPT = """现在是夜晚，你需要决定是否使用解药和毒药。
今晚被狼人击杀的玩家是：{tonight_kill}
你的解药状态：{antidote_status}
你的毒药状态：{poison_status}

请输出 JSON：{{"action": "witch_action", "save_target": "Player_X或none", "poison_target": "Player_Y或none"}}
save_target: 要救的玩家ID，或 "none" 表示不救
poison_target: 要毒杀的玩家ID，或 "none" 表示不毒"""

VOTE_PROMPT = """现在是投票阶段，你需要投票给一名玩家。
根据你的信息分析，投票给你认为最可能是狼人的玩家。
请输出 JSON：{"action": "vote", "target": "Player_X", "reason": "你的理由"}
其中 Player_X 替换为具体的玩家ID，或 "skip" 表示弃票。"""

SPEECH_PROMPT = """现在是白天讨论阶段，你需要公开发言。
你是女巫，要小心不要暴露身份，但可以根据掌握的信息给出有价值的分析。
请直接输出你的发言内容（纯文本，不要 JSON）。"""

MONOLOGUE_PROMPT = """在发言之前，先进行内心独白分析。
分析：解药和毒药的使用策略？谁可能是狼人？如何保护自己？
请直接输出你的内心分析（纯文本，不要 JSON）。"""


class WitchAgent(BaseAgent):
    def __init__(self, player_id: str):
        super().__init__(player_id, Role.WITCH)

    def _build_system_prompt(self) -> str:
        return WITCH_SYSTEM_PROMPT

    def _fallback_night_action(self, game_info: dict) -> dict:
        save_target = "none"
        poison_target = "none"

        tonight_kill = game_info.get("tonight_kill")
        if tonight_kill and not game_info.get("antidote_used", False):
            save_target = tonight_kill

        alive_others = self.get_alive_others(game_info)
        if not game_info.get("poison_used", False) and alive_others and random.random() < 0.3:
            poison_target = random.choice(alive_others)

        return {"action": "witch_action", "save_target": save_target, "poison_target": poison_target}

    def decide_night_action(self, game_info: dict) -> dict:
        if not self.is_llm_enabled():
            return self._fallback_night_action(game_info)

        context = build_game_context(game_info, self.memory.inject_to_prompt())
        tonight_kill = game_info.get("tonight_kill", "无")
        antidote_status = "已使用" if game_info.get("antidote_used") else "未使用"
        poison_status = "已使用" if game_info.get("poison_used") else "未使用"

        user_prompt = context + "\n\n" + NIGHT_PROMPT.format(
            tonight_kill=tonight_kill,
            antidote_status=antidote_status,
            poison_status=poison_status,
        )

        result = call_llm_json(self.system_prompt, user_prompt, temperature=0.5)
        if result and "save_target" in result and "poison_target" in result:
            return {
                "action": "witch_action",
                "save_target": result["save_target"],
                "poison_target": result["poison_target"],
            }
        return self._fallback_night_action(game_info)

    def _fallback_vote(self, game_info: dict) -> dict:
        alive_others = self.get_alive_others(game_info)
        target = random.choice(alive_others) if alive_others else "skip"
        return {"action": "vote", "target": target, "reason": "根据场上信息判断"}

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
        return f"我是{self.player_id}，我觉得大家应该多关注发言中的逻辑漏洞，而不是急于互相指责。"

    def generate_speech(self, game_info: dict) -> str:
        if not self.is_llm_enabled():
            return self._fallback_speech(game_info)

        context = build_game_context(game_info, self.memory.inject_to_prompt())
        user_prompt = f"{context}\n\n{SPEECH_PROMPT}"

        result = call_llm(self.system_prompt, user_prompt, temperature=0.8)
        return result if result else self._fallback_speech(game_info)

    def _fallback_monologue(self, game_info: dict) -> str:
        antidote_used = "已用" if game_info.get("antidote_used") else "未用"
        poison_used = "已用" if game_info.get("poison_used") else "未用"
        return f"我是女巫，解药{antidote_used}，毒药{poison_used}。需要谨慎选择用药时机。"

    def inner_monologue(self, game_info: dict) -> str:
        if not self.is_llm_enabled():
            return self._fallback_monologue(game_info)

        context = build_game_context(game_info, self.memory.inject_to_prompt())
        user_prompt = f"{context}\n\n{MONOLOGUE_PROMPT}"

        result = call_llm(self.system_prompt, user_prompt, temperature=0.7)
        return result if result else self._fallback_monologue(game_info)
