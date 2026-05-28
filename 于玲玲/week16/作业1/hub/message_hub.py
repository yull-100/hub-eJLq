"""消息中枢 - 公共广播与私密频道"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class MsgType(Enum):
    PUBLIC = "public"
    PRIVATE = "private"
    GROUP = "group"
    SYSTEM = "system"


@dataclass
class Message:
    sender: str
    content: str
    msg_type: MsgType
    round: int = 0
    phase: str = ""
    target: Optional[str] = None
    group: Optional[str] = None


class MessageHub:
    def __init__(self):
        self.messages: List[Message] = []
        self.groups: Dict[str, List[str]] = {
            "werewolf": [],
        }
        self.current_round: int = 0
        self.current_phase: str = ""

    def new_round(self, round_num: int, phase):
        self.current_round = round_num
        self.current_phase = phase.value if hasattr(phase, "value") else str(phase)

    def set_group_members(self, group_name: str, members: List[str]):
        self.groups[group_name] = members

    def broadcast(self, sender: str, content: str):
        msg = Message(
            sender=sender,
            content=content,
            msg_type=MsgType.PUBLIC,
            round=self.current_round,
            phase=self.current_phase,
        )
        self.messages.append(msg)

    def send_private(self, sender: str, target: str, content: str):
        msg = Message(
            sender=sender,
            content=content,
            msg_type=MsgType.PRIVATE,
            target=target,
            round=self.current_round,
            phase=self.current_phase,
        )
        self.messages.append(msg)

    def send_group(self, sender: str, group_name: str, content: str):
        msg = Message(
            sender=sender,
            content=content,
            msg_type=MsgType.GROUP,
            group=group_name,
            round=self.current_round,
            phase=self.current_phase,
        )
        self.messages.append(msg)

    def get_public_messages(self, round_num: Optional[int] = None) -> List[Message]:
        result = [m for m in self.messages if m.msg_type == MsgType.PUBLIC]
        if round_num is not None:
            result = [m for m in result if m.round == round_num]
        return result

    def get_private_messages(self, target: str, round_num: Optional[int] = None) -> List[Message]:
        result = [
            m for m in self.messages
            if m.msg_type == MsgType.PRIVATE and m.target == target
        ]
        if round_num is not None:
            result = [m for m in result if m.round == round_num]
        return result

    def get_group_messages(self, group_name: str, round_num: Optional[int] = None) -> List[Message]:
        result = [
            m for m in self.messages
            if m.msg_type == MsgType.GROUP and m.group == group_name
        ]
        if round_num is not None:
            result = [m for m in result if m.round == round_num]
        return result

    def get_visible_messages(self, player_id: str, player_camp: str, round_num: Optional[int] = None) -> List[Message]:
        """根据玩家身份获取其可见的消息（信息脱敏）"""
        visible = []
        for msg in self.messages:
            if msg.msg_type == MsgType.PUBLIC:
                visible.append(msg)
            elif msg.msg_type == MsgType.SYSTEM:
                visible.append(msg)
            elif msg.msg_type == MsgType.PRIVATE and msg.target == player_id:
                visible.append(msg)
            elif msg.msg_type == MsgType.GROUP:
                if msg.group == "werewolf" and player_camp == "evil":
                    visible.append(msg)
        if round_num is not None:
            visible = [m for m in visible if m.round == round_num]
        return visible
