"""记忆模块 - 管理 Agent 的对话历史和结构化记忆"""
from typing import Dict, List, Optional, Any


class Memory:
    def __init__(self, player_id: str):
        self.player_id = player_id
        self.conversation_history: List[dict] = []
        self.vote_records: Dict[int, Dict[str, str]] = {}
        self.identity_claims: Dict[str, str] = {}
        self.private_info: List[dict] = []
        self.events: List[dict] = []

    def add_conversation(self, round_num: int, speaker: str, content: str):
        self.conversation_history.append({
            "round": round_num,
            "speaker": speaker,
            "content": content,
        })

    def add_vote_record(self, round_num: int, votes: Dict[str, str]):
        self.vote_records[round_num] = votes

    def add_identity_claim(self, player_id: str, claimed_role: str):
        self.identity_claims[player_id] = claimed_role

    def add_private_info(self, round_num: int, info_type: str, content: Any):
        self.private_info.append({
            "round": round_num,
            "type": info_type,
            "content": content,
        })

    def add_event(self, round_num: int, event_type: str, detail: str):
        self.events.append({
            "round": round_num,
            "type": event_type,
            "detail": detail,
        })

    def get_round_summary(self, round_num: int) -> str:
        lines = [f"=== 第 {round_num} 轮 ==="]

        convs = [c for c in self.conversation_history if c["round"] == round_num]
        if convs:
            lines.append("发言：")
            for c in convs:
                lines.append(f"  {c['speaker']}: {c['content']}")

        if round_num in self.vote_records:
            lines.append("投票：")
            for voter, target in self.vote_records[round_num].items():
                lines.append(f"  {voter} -> {target}")

        deaths = [e for e in self.events if e["round"] == round_num and e["type"] == "death"]
        if deaths:
            lines.append("死亡：")
            for d in deaths:
                lines.append(f"  {d['detail']}")

        return "\n".join(lines)

    def get_full_summary(self) -> str:
        if not self.conversation_history and not self.events:
            return "暂无历史记录。"

        max_round = max(
            [c["round"] for c in self.conversation_history] +
            [e["round"] for e in self.events] +
            list(self.vote_records.keys()) +
            [0]
        )

        summaries = []
        for r in range(1, max_round + 1):
            summary = self.get_round_summary(r)
            if summary:
                summaries.append(summary)

        return "\n\n".join(summaries) if summaries else "暂无历史记录。"

    def inject_to_prompt(self) -> str:
        """生成可注入 Prompt 的历史摘要"""
        parts = []

        if self.identity_claims:
            parts.append("【身份声明】")
            for pid, role in self.identity_claims.items():
                parts.append(f"  {pid} 自称 {role}")

        if self.private_info:
            parts.append("【私密信息】")
            for info in self.private_info:
                parts.append(f"  第{info['round']}轮: {info['content']}")

        history = self.get_full_summary()
        if history != "暂无历史记录。":
            parts.append("【历史记录】")
            parts.append(history)

        return "\n".join(parts) if parts else "暂无历史信息。"
