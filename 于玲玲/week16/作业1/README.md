# 狼人杀多智能体博弈系统

基于大语言模型的多智能体狼人杀博弈系统。多个 LLM Agent 分别扮演狼人、预言家、女巫、村民等角色，进行自主博弈。

## 项目特性

- 三层解耦架构：游戏引擎层、消息中枢层、智能体层
- 支持 OpenAI 兼容 API（GPT-4、DeepSeek、本地模型等）
- 结构化决策：Agent 输出 JSON，引擎校验合法性
- 心智理论（ToM）：Agent 生成内心独白分析局势
- 记忆系统：跨轮次历史记录注入 Prompt
- 信息脱敏：根据阵营分发不同视角信息

## 项目结构

```
.
├── main.py                    # 入口文件（CLI / HTTP 服务器）
├── requirements.txt           # 依赖列表
├── 实现思路.txt               # 设计文档
│
├── config/                    # 配置层
│   ├── __init__.py
│   └── settings.py            # 全局配置（游戏参数、LLM配置、动作Schema）
│
├── engine/                    # 游戏引擎层
│   ├── __init__.py
│   ├── roles.py               # 角色/阵营/阶段枚举
│   ├── game_state.py          # 游戏状态管理
│   └── game_engine.py         # 状态机引擎（驱动游戏流程）
│
├── hub/                       # 消息中枢层
│   ├── __init__.py
│   └── message_hub.py         # 消息路由（公共/私密/组消息）
│
├── api/                       # HTTP API 层（FastAPI）
│   ├── __init__.py
│   ├── app.py                 # FastAPI 应用与路由
│   └── game_manager.py        # 游戏实例管理器
│
├── fronted/                   # Vue 3 前端页面
│   ├── index.html             # 主页面
│   ├── css/
│   │   └── style.css          # 样式
│   └── js/
│       └── app.js             # Vue 应用逻辑
│
└── agents/                    # 智能体层
    ├── __init__.py
    ├── memory.py              # 记忆模块
    ├── base_agent.py          # Agent基类
    ├── llm_client.py          # LLM客户端封装
    ├── mock_agent.py          # Mock随机Agent（测试用）
    ├── werewolf_agent.py      # 狼人Agent
    ├── seer_agent.py          # 预言家Agent
    ├── witch_agent.py         # 女巫Agent
    └── villager_agent.py      # 村民Agent
```

## 安装与运行

### 1. 安装依赖

```bash
# Mock 模式无需安装依赖（随机策略）
# LLM 模式需要安装 openai
pip install -r requirements.txt
```

### 2. 运行游戏

#### Mock 模式（无需 API Key）

```bash
python main.py
```

使用随机策略运行，适合快速测试流程。

#### LLM 模式

```bash
# 使用 OpenAI API
python main.py --llm --api-key sk-xxx --model gpt-4o-mini

# 使用 DeepSeek API
python main.py --llm --api-key sk-xxx --model deepseek-chat --base-url https://api.deepseek.com/v1

# 使用环境变量
export OPENAI_API_KEY=sk-xxx
python main.py --llm --model gpt-4o-mini
```

#### 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--llm` | 启用 LLM 模式 | 禁用 |
| `--api-key` | OpenAI API Key | 无 |
| `--model` | 模型名称 | `gpt-4o-mini` |
| `--base-url` | API Base URL（兼容第三方服务） | OpenAI 官方 |
| `--rounds` | 最大回合数 | `10` |

### 3. HTTP 服务器模式（为前端提供 API）

```bash
# Mock 模式启动（随机策略，无需 API Key）
python main.py --server

# LLM 模式启动
python main.py --server --llm

# 开发模式（热重载）
python main.py --server --reload

# 自定义端口
python main.py --server --host 0.0.0.0 --port 8080
```

启动后访问：
- 前端页面：`http://localhost:8000/`
- API 根路径：`http://localhost:8000/api/config`
- Swagger 交互式文档：`http://localhost:8000/docs`

#### 服务器模式参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--server` | 以 HTTP 服务器模式启动 | 禁用 |
| `--host` | 监听地址 | `0.0.0.0` |
| `--port` | 监听端口 | `8000` |
| `--reload` | 热重载（开发用） | 禁用 |

### 4. 配置 LLM（可选）

也可以在 `config/settings.py` 中直接修改配置：

```python
LLM_CONFIG = {
    "enabled": True,
    "model_name": "gpt-4o-mini",
    "api_key": "sk-xxx",
    "base_url": "",  # 留空使用 OpenAI 默认地址
    "temperature": 0.7,
    "max_tokens": 1024,
}
```

或使用快捷函数：

```python
from config.settings import enable_llm, disable_llm

# 启用 LLM
enable_llm(api_key="sk-xxx", model_name="gpt-4o-mini")

# 禁用 LLM（回退到 Mock 模式）
disable_llm()
```

## 游戏规则

### 角色配置（5人局）

| 角色 | 数量 | 阵营 | 能力 |
|------|------|------|------|
| 狼人 | 2 | 邪恶 | 夜间击杀一名玩家 |
| 预言家 | 1 | 善良 | 夜间查验一名玩家身份 |
| 女巫 | 1 | 善良 | 拥有一瓶解药和一瓶毒药（各一次） |
| 村民 | 1 | 善良 | 无特殊能力 |

### 游戏流程

1. **夜晚阶段**
   - 狼人商议并选择击杀目标
   - 预言家查验一名玩家身份
   - 女巫决定是否使用解药（救人）和毒药（毒人）

2. **白天讨论阶段**
   - 所有存活玩家按顺序发言
   - 可以分析局势、互相质疑、声明身份

3. **白天投票阶段**
   - 所有存活玩家投票
   - 得票最多的玩家被淘汰

4. **胜负判定**
   - **好人阵营胜利**：所有狼人被淘汰
   - **狼人阵营胜利**：狼人数量 ≥ 好人数量

## 架构设计

### 三层解耦

```
┌─────────────────────────────────────────┐
│  游戏引擎层 (Game Engine)               │
│  - 维护全局状态                          │
│  - 驱动状态机流转                        │
│  - 校验动作合法性                        │
│  - 信息脱敏分发                          │
└─────────────────────────────────────────┘
              ↕
┌─────────────────────────────────────────┐
│  消息中枢层 (Message Hub)               │
│  - 公共广播（白天发言）                  │
│  - 私密消息（预言家查验结果）            │
│  - 组消息（狼人夜间交流）                │
└─────────────────────────────────────────┘
              ↕
┌─────────────────────────────────────────┐
│  智能体层 (Agent)                       │
│  - 角色专属 System Prompt               │
│  - 私有记忆模块                          │
│  - 结构化 JSON 输出                      │
│  - 心智理论内心独白                      │
└─────────────────────────────────────────┘
```

### 核心机制

#### 1. 结构化决策

Agent 所有动作强制输出 JSON，引擎校验：

```json
// 投票
{"action": "vote", "target": "Player_3", "reason": "发言逻辑有漏洞"}

// 狼人击杀
{"action": "kill", "target": "Player_2"}

// 预言家查验
{"action": "check", "target": "Player_4"}

// 女巫用药
{"action": "witch_action", "save_target": "Player_1", "poison_target": "none"}
```

#### 2. 记忆与历史注入

Agent 维护两类记忆：
- **对话历史**：所有公开发言记录
- **结构化记忆**：投票记录表、身份声明表、死亡事件

每轮决策时，将记忆摘要动态注入 Prompt，保证跨轮次推理能力。

#### 3. 心智理论（ToM）

Agent 在发言前生成**不对外广播**的内心独白：

```
[Player_2 内心] 我是狼人，队友是['Player_4']。
需要隐藏身份，找机会击杀关键角色。
Player_1 的发言太有逻辑了，可能是预言家，今晚要优先击杀。

[Player_2 发言] 大家好，我是Player_2，我觉得我们应该仔细分析...
```

这种"内心戏"提升了 Agent 的策略性和欺骗能力。

#### 4. 信息脱敏

引擎根据玩家阵营分发不同视角的信息：
- **狼人**：知道队友身份，不知道其他角色
- **好人**：只知道自己的身份
- **预言家**：通过查验逐步获取信息
- **女巫**：知道每晚谁被击杀

## HTTP API 接口文档（FastAPI）

系统提供 HTTP API 供前端或其他服务调用，可逐阶段推进游戏并获取结构化结果。

### 启动服务器

```bash
# Mock 模式
python main.py --server

# 指定端口
python main.py --server --port 8080

# 开发模式（热重载）
python main.py --server --reload
```

访问 `http://localhost:8000/docs` 查看 Swagger 交互式文档。

### 接口列表

#### 游戏管理

| 方法 | 路径 | 说明 |
|------|------|------|
| **POST** | `/api/game` | 创建新游戏，返回 `game_id` |
| **GET** | `/api/games` | 获取所有游戏实例列表 |
| **GET** | `/api/game/{game_id}` | 获取游戏状态（结束前不暴露角色身份） |
| **DELETE** | `/api/game/{game_id}` | 删除游戏实例 |

创建游戏时可选参数（JSON body）：
```json
{"player_count": 5, "max_rounds": 10}
```

#### 游戏流程控制

| 方法 | 路径 | 说明 |
|------|------|------|
| **POST** | `/api/game/{game_id}/step` | **单步执行**：按 夜晚→白天讨论→白天投票 顺序推进一个阶段 |
| **POST** | `/api/game/{game_id}/run` | **完整运行**：直接运行整局游戏至结束 |

`step` 接口每调用一次前进一个阶段：
```
    创建游戏 → 夜晚 → 白天讨论 → 白天投票 → 夜晚 → ... → 游戏结束
    （第1步）  （第2步）  （第3步）   （第4步）  （第5步）
```

`step` 返回结果示例（夜晚阶段）：
```json
{
  "step_result": {
    "phase": "night",
    "round": 1,
    "winner": null,
    "data": {
      "kill_target": "Player_3",
      "check_result": {
        "Player_2": {"target": "Player_3", "result": "好人"}
      },
      "save_target": "Player_3",
      "poison_target": "none",
      "killed": []
    }
  },
  "state": {
    "game_id": "game_1",
    "round": 1,
    "phase": "day_discuss",
    "winner": null,
    "alive_count": 5,
    "player_count": 5
  }
}
```

#### 消息与日志

| 方法 | 路径 | 说明 |
|------|------|------|
| **GET** | `/api/game/{game_id}/messages` | 获取消息记录（可选 `?player_id=Player_1` 按角色脱敏） |
| **GET** | `/api/game/{game_id}/logs` | 获取游戏日志 |

#### 其他

| 方法 | 路径 | 说明 |
|------|------|------|
| **GET** | `/api/config` | 获取当前游戏配置 |

### 前端集成建议

1. **创建游戏** → `POST /api/game` 获取 `game_id`
2. **轮询状态** → 定时 `GET /api/game/{game_id}` 显示当前阶段和存活玩家
3. **驱动流程** → 用户点击"下一步"时调用 `POST /api/game/{game_id}/step`
4. **显示消息** → `GET /api/game/{game_id}/messages?player_id=...` 按角色视角展示
5. **角色查看** → 游戏结束后 `GET /api/game/{game_id}` 返回全员身份

## 扩展开发

### 新增角色

1. 在 `engine/roles.py` 添加角色枚举
2. 在 `config/settings.py` 添加角色配置
3. 创建 `agents/xxx_agent.py`，继承 `BaseAgent`
4. 实现四个抽象方法：
   - `_build_system_prompt()`
   - `decide_night_action(game_info)`
   - `decide_vote(game_info)`
   - `generate_speech(game_info)`
   - `inner_monologue(game_info)`
5. 在 `agents/mock_agent.py` 的 `create_agent_for_role()` 中注册

### 接入其他 LLM

修改 `agents/llm_client.py`，或创建新的客户端模块。只要实现两个函数：
- `call_llm(system_prompt, user_prompt)` → 返回文本
- `call_llm_json(system_prompt, user_prompt)` → 返回解析后的 dict

### 调整游戏配置

修改 `config/settings.py` 中的 `GAME_CONFIG`：

```python
GAME_CONFIG = {
    "player_count": 8,  # 修改总人数
    "roles": {
        "werewolf": 2,
        "seer": 1,
        "witch": 1,
        "hunter": 1,  # 新增猎人
        "villager": 3,
    },
    "max_rounds": 15,
}
```

## 进阶方向

| 方向 | 核心逻辑 |
|------|---------|
| 自演化 Agent | 元智能体赛后读取决策日志 → 分析失败原因 → 自动优化 System Prompt → 热更新投入下一局 |
| 多维评测体系 | 结果评测（胜率/存活轮数）+ 过程评测（Judge Agent 打分：逻辑自洽性、煽动性、信息贡献度）+ 复盘归因报告 + Leaderboard |
| 自进化闭环 | 自动化竞技场持续自我对弈，用进化策略筛选最优 Prompt 和思维链模式 |

## 开发约定

- 文件读写仅限当前工作目录
- 游戏流程状态机是系统的核心控制流，修改时需确保所有状态转换路径完整覆盖
- 新增角色时，需同时定义：System Prompt、Action Space、所属消息频道（公开/私密/组）
- Agent 的所有对外动作必须经过引擎层的合法性校验，不允许 Agent 直接修改 GameState

## License

MIT
