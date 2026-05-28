# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

基于大模型的多智能体狼人杀博弈系统。多个 LLM Agent 分别扮演狼人、预言家、女巫、村民等角色，进行自主博弈。

## 构建与运行

```bash
# 安装依赖（LLM 模式需要）
pip install -r requirements.txt

# Mock 模式运行（无需 API Key，随机策略）
python main.py

# LLM 模式运行
python main.py --llm --api-key YOUR_KEY --model gpt-4o-mini

# 使用 OpenAI 兼容的第三方 API（如 DeepSeek）
python main.py --llm --api-key YOUR_KEY --model deepseek-chat --base-url https://api.deepseek.com/v1

# 指定最大回合数
python main.py --rounds 5
```

也可通过设置环境变量 `OPENAI_API_KEY` 来代替 `--api-key` 参数。

## 架构设计（三层解耦）

### 游戏引擎层（Game Engine）

- 维护全局 `GameState`，通过状态机驱动 `夜晚 → 白天发言 → 投票 → 结算` 的回合流转
- 充当裁判：接收 Agent 决策，校验合法性后更新全局状态
- **信息脱敏**：根据角色阵营分发不同视角的信息（如死因是否公开由女巫决定是否告知）

### 消息中枢层（Message Hub）

- **公共聊天室**：白天发言通过消息总线广播给所有存活玩家
- **私密频道**：狼人夜间交流走狼人组 Group 消息，预言家查验结果走点对点消息，确保信息隔离

### 智能体层（Agent）

- 每个 Agent 拥有独立的 System Prompt、私有 Memory 和 Action Space
- 狼人 Agent 侧重伪装与夜间协作；预言家 Agent 侧重逻辑引导与风险规避
- Agent 通过结构化 JSON 输出决策，引擎通过 JSON Schema 校验并执行

## 核心机制

### LLM 集成

- 使用 OpenAI SDK（兼容任何 OpenAI 格式的 API）
- `agents/llm_client.py` 封装了 `call_llm()` 和 `call_llm_json()` 两个调用函数
- 各角色 Agent 内置随机策略回退：LLM 未启用或调用失败时自动使用随机决策
- LLM 配置在 `config/settings.py` 的 `LLM_CONFIG`，或通过命令行参数传入

### 结构化决策

Agent 所有动作强制输出结构化 JSON，例如：
```json
{"action": "vote", "target": "Player_3", "reason": "..."}
```
引擎侧用 JSON Schema 校验，非法输出要求重试。

### 记忆与历史注入

- 维护两类记忆：**对话历史**（原始发言）+ **结构化记忆**（投票记录表、身份声明表等）
- 每轮 Prompt 动态注入关键历史摘要，避免上下文溢出，同时保证跨轮次推理能力

### 心智理论（ToM）引导

Agent 发言前先生成不对外广播的内心独白（Chain of Thought），分析对手意图和自身伪装策略，再产出公开发言。

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
