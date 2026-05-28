"""Mock Agent - 随机决策用于流程测试"""
import random
from typing import List

from agents.base_agent import BaseAgent
from engine.roles import Role


class MockAgent(BaseAgent):
    def _build_system_prompt(self) -> str:
        return f"你是 {self.player_id}，正在参与一场狼人杀游戏。当前使用随机策略进行测试。"

    def decide_night_action(self, game_info: dict) -> dict:
        role = game_info.get("my_role", "")
        alive_others = self.get_alive_others(game_info)
        known_roles = game_info.get("known_roles", {})

        if role == "werewolf":
            targets = [p for p in alive_others if known_roles.get(p) != "werewolf"]
            target = random.choice(targets) if targets else alive_others[0]
            return {"action": "kill", "target": target}

        elif role == "seer":
            target = random.choice(alive_others) if alive_others else self.player_id
            return {"action": "check", "target": target}

        elif role == "witch":
            save_target = "none"
            poison_target = "none"
            tonight_kill = game_info.get("tonight_kill")
            if tonight_kill and not game_info.get("antidote_used", False):
                save_target = tonight_kill
            if not game_info.get("poison_used", False) and alive_others and random.random() < 0.3:
                poison_target = random.choice(alive_others)
            return {"action": "witch_action", "save_target": save_target, "poison_target": poison_target}

        return {"action": "none"}

    def decide_vote(self, game_info: dict) -> dict:
        alive_others = self.get_alive_others(game_info)
        if alive_others:
            target = random.choice(alive_others)
        else:
            target = "skip"
        return {"action": "vote", "target": target, "reason": "随机投票"}

    def generate_speech(self, game_info: dict) -> str:
        role_cn = {
            "werewolf": "狼人", "seer": "预言家",
            "witch": "女巫", "villager": "村民",
        }
        my_role = game_info.get("my_role", "unknown")
        # 狼人不暴露身份，假装村民
        if my_role == "werewolf":
            claim = "村民"
        else:
            claim = role_cn.get(my_role, "村民")
        return f"大家好，我是{self.player_id}，我自称是{claim}。目前我还在观察大家的发言，暂时没有特别的怀疑对象。"

    def inner_monologue(self, game_info: dict) -> str:
        role = game_info.get("my_role", "")
        alive_others = self.get_alive_others(game_info)
        if role == "werewolf":
            return f"我是狼人，队友是{self.teammates}。今晚要选一个有价值的目标击杀。"
        elif role == "seer":
            return f"我是预言家，需要谨慎查验。目前存活玩家：{alive_others}。"
        return f"我是好人阵营，需要仔细观察发言寻找破绽。"


def create_agent_for_role(player_id: str, role: Role) -> BaseAgent:
    """根据角色创建对应的 Agent（含 LLM 集成，LLM 未启用时自动回退到随机策略）"""
    from agents.werewolf_agent import WerewolfAgent
    from agents.seer_agent import SeerAgent
    from agents.witch_agent import WitchAgent
    from agents.villager_agent import VillagerAgent

    agent_map = {
        Role.WEREWOLF: WerewolfAgent,
        Role.SEER: SeerAgent,
        Role.WITCH: WitchAgent,
        Role.VILLAGER: VillagerAgent,
    }

    agent_cls = agent_map.get(role)
    if agent_cls:
        return agent_cls(player_id)
    return MockAgent(player_id, role)
