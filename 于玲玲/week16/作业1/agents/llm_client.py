"""LLM 客户端 - 封装 OpenAI API 调用"""
import json
from typing import Optional

from config.settings import LLM_CONFIG

_client = None


def get_client():
    """获取 OpenAI 客户端（单例）"""
    global _client
    if _client is None:
        try:
            from openai import OpenAI
            _client = OpenAI(
                api_key=LLM_CONFIG.get("api_key", ""),
                base_url=LLM_CONFIG.get("base_url", None),
            )
        except ImportError:
            print("警告：未安装 openai 库，请运行 pip install openai")
            return None
    return _client


def call_llm(
    system_prompt: str,
    user_prompt: str,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> str:
    """调用 LLM 并返回文本响应"""
    if not LLM_CONFIG.get("enabled", False):
        return ""

    client = get_client()
    if not client:
        return ""

    try:
        response = client.chat.completions.create(
            model=LLM_CONFIG.get("model_name", "gpt-3.5-turbo"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature or LLM_CONFIG.get("temperature", 0.7),
            max_tokens=max_tokens or LLM_CONFIG.get("max_tokens", 1024),
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"LLM 调用失败: {e}")
        return ""


def call_llm_json(
    system_prompt: str,
    user_prompt: str,
    temperature: Optional[float] = None,
) -> Optional[dict]:
    """调用 LLM 并解析 JSON 响应"""
    raw = call_llm(system_prompt, user_prompt, temperature)
    if not raw:
        return None

    # 尝试提取 JSON（支持 markdown 代码块包裹）
    text = raw.strip()
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        print(f"LLM 返回非 JSON: {raw[:200]}...")
        return None


def build_game_context(game_info: dict, memory_summary: str) -> str:
    """构建游戏上下文 prompt"""
    parts = [
        f"当前轮次：第 {game_info.get('round', 0)} 轮",
        f"当前阶段：{game_info.get('phase', '')}",
        f"你的身份：{game_info.get('my_id', '')}（{game_info.get('my_role', '')}）",
        f"你的阵营：{game_info.get('my_camp', '')}",
        "",
        "存活玩家：",
    ]

    for p in game_info.get("alive_players", []):
        status = "存活" if p["is_alive"] else "死亡"
        parts.append(f"  {p['player_id']}: {status}")

    known = game_info.get("known_roles", {})
    if known:
        parts.append("")
        parts.append("已知身份：")
        for pid, role in known.items():
            parts.append(f"  {pid}: {role}")

    deaths = game_info.get("deaths", [])
    if deaths:
        parts.append("")
        parts.append("死亡记录：")
        for d in deaths:
            parts.append(f"  第{d['round']}轮: {d['player_id']} 死亡")

    parts.append("")
    parts.append("历史记录：")
    parts.append(memory_summary)

    return "\n".join(parts)
