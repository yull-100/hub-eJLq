"""狼人杀多智能体博弈系统 - 入口"""
import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine.game_engine import GameEngine
from config.settings import GAME_CONFIG, enable_llm


def run_cli(args):
    """命令行模式：直接运行一局游戏"""
    config = GAME_CONFIG.copy()
    config["max_rounds"] = args.rounds

    if args.llm or args.api_key:
        api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            print("错误：启用 LLM 需要提供 --api-key 或设置环境变量 OPENAI_API_KEY")
            print("当前将以 Mock（随机策略）模式运行。")
        else:
            enable_llm(api_key=api_key, model_name=args.model, base_url=args.base_url)
            print(f"LLM 已启用：模型={args.model}")

    engine = GameEngine(config)
    engine.run()


def run_server(args):
    """服务器模式：启动 FastAPI HTTP 服务"""
    if args.api_key:
        enable_llm(api_key=args.api_key, model_name=args.model, base_url=args.base_url)
    elif args.llm:
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if api_key:
            enable_llm(api_key=api_key, model_name=args.model, base_url=args.base_url)

    import uvicorn
    host = args.host
    port = args.port
    print(f"启动 HTTP 服务：http://{host}:{port}")
    print(f"API 文档：http://{host}:{port}/docs")
    uvicorn.run("api.app:app", host=host, port=port, reload=args.reload)


def main():
    parser = argparse.ArgumentParser(description="狼人杀多智能体博弈系统")
    parser.add_argument("--llm", action="store_true", help="启用 LLM 模式")
    parser.add_argument("--api-key", type=str, default="", help="OpenAI API Key")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="模型名称")
    parser.add_argument("--base-url", type=str, default="", help="API Base URL（兼容 OpenAI 格式的第三方服务）")
    parser.add_argument("--rounds", type=int, default=10, help="最大回合数（CLI 模式）")

    # 服务器模式参数
    parser.add_argument("--server", action="store_true", help="以 HTTP 服务器模式启动")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="服务器监听地址（默认 0.0.0.0）")
    parser.add_argument("--port", type=int, default=8000, help="服务器端口（默认 8000）")
    parser.add_argument("--reload", action="store_true", help="热重载（开发用）")

    args = parser.parse_args()

    if args.server:
        run_server(args)
    else:
        run_cli(args)


if __name__ == "__main__":
    main()
