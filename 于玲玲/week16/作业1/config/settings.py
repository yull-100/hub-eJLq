"""全局配置"""

# ========== 游戏配置 ==========
GAME_CONFIG = {
    "player_count": 5,
    "roles": {
        "werewolf": 2,
        "seer": 1,
        "witch": 1,
        "villager": 1,
    },
    "max_rounds": 10,
}

# ========== LLM 配置（OpenAI 兼容） ==========
# 启用方式：设置 enabled=True 并填入 api_key
# 支持任何 OpenAI 兼容的 API（DeepSeek、本地 Ollama 等），修改 base_url 即可
LLM_CONFIG = {
    "enabled": True,
    "model_name": "qwen3.7-max-2026-05-17",
    # "api_key": "sk-20fff46b235d46de919283b69eca670f",
    "api_key": "sk-0080a0ba093043c28d3536fcaa33ac2f",
    # "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",  # 留空使用 OpenAI 默认地址；例如 "https://api.deepseek.com/v1"
    "base_url": "https://api.deepseek.com",
    "temperature": 0.7,
    "max_tokens": 1024,
}


def enable_llm(api_key: str, model_name: str = "gpt-4o-mini", base_url: str = ""):
    """快捷启用 LLM"""
    LLM_CONFIG["enabled"] = True
    LLM_CONFIG["api_key"] = api_key
    LLM_CONFIG["model_name"] = model_name
    LLM_CONFIG["base_url"] = base_url


def disable_llm():
    """禁用 LLM（回退到 Mock 随机策略）"""
    LLM_CONFIG["enabled"] = False

# ========== 动作 Schema ==========
ACTION_SCHEMAS = {
    "kill": {
        "required": ["action", "target"],
        "properties": {
            "action": {"const": "kill"},
            "target": {"type": "string", "pattern": "^Player_\\d+$"},
        },
    },
    "check": {
        "required": ["action", "target"],
        "properties": {
            "action": {"const": "check"},
            "target": {"type": "string", "pattern": "^Player_\\d+$"},
        },
    },
    "save": {
        "required": ["action", "target"],
        "properties": {
            "action": {"const": "save"},
            "target": {"type": "string", "pattern": "^Player_\\d+$|^none$"},
        },
    },
    "poison": {
        "required": ["action", "target"],
        "properties": {
            "action": {"const": "poison"},
            "target": {"type": "string", "pattern": "^Player_\\d+$|^none$"},
        },
    },
    "vote": {
        "required": ["action", "target", "reason"],
        "properties": {
            "action": {"const": "vote"},
            "target": {"type": "string", "pattern": "^Player_\\d+$|^skip$"},
            "reason": {"type": "string"},
        },
    },
    "speak": {
        "required": ["action", "content"],
        "properties": {
            "action": {"const": "speak"},
            "content": {"type": "string"},
        },
    },
    "witch_action": {
        "required": ["action", "save_target", "poison_target"],
        "properties": {
            "action": {"const": "witch_action"},
            "save_target": {"type": "string", "pattern": "^Player_\\d+$|^none$"},
            "poison_target": {"type": "string", "pattern": "^Player_\\d+$|^none$"},
        },
    },
}


def validate_action(action_dict: dict, schema_name: str) -> bool:
    """校验 Agent 动作是否符合 Schema"""
    schema = ACTION_SCHEMAS.get(schema_name)
    if not schema:
        return False
    for key in schema["required"]:
        if key not in action_dict:
            return False
    for key, rule in schema["properties"].items():
        if key in action_dict:
            value = action_dict[key]
            if "const" in rule and value != rule["const"]:
                return False
            if "type" in rule:
                if rule["type"] == "string" and not isinstance(value, str):
                    return False
            if "pattern" in rule:
                import re
                if not re.match(rule["pattern"], value):
                    return False
    return True
