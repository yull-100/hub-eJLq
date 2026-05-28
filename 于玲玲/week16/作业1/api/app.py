"""FastAPI 应用 - 狼人杀 HTTP API 接口"""
import os
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional

from api.game_manager import game_manager
from config.settings import GAME_CONFIG

app = FastAPI(
    title="狼人杀多智能体博弈系统 API",
    description="HTTP 接口，供前端调用游戏引擎",
    version="1.0.0",
)

# 允许跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 前端文件目录
_fronted_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "fronted")


def _read_fronted_file(path: str) -> Optional[str]:
    """读取前端静态文件"""
    full_path = os.path.join(_fronted_dir, path)
    if os.path.isfile(full_path):
        with open(full_path, "r", encoding="utf-8") as f:
            return f.read()
    return None


# ──────────────────────────── 请求/响应模型 ────────────────────────────

class CreateGameRequest(BaseModel):
    player_count: Optional[int] = Field(None, description="玩家数量（覆盖配置）")
    max_rounds: Optional[int] = Field(None, description="最大轮数（覆盖配置）")


# ──────────────────────────── 游戏管理 ────────────────────────────

@app.post("/api/game", summary="创建新游戏")
def create_game(req: CreateGameRequest = None):
    """创建新游戏实例，返回 game_id"""
    config = None
    if req:
        config = GAME_CONFIG.copy()
        if req.player_count:
            config["player_count"] = req.player_count
        if req.max_rounds:
            config["max_rounds"] = req.max_rounds

    game_id = game_manager.create_game(config)
    state = game_manager.get_game_state(game_id)
    return {"game_id": game_id, "state": state}


@app.get("/api/games", summary="获取所有游戏列表")
def list_games():
    return {"games": game_manager.list_games()}


@app.get("/api/game/{game_id}", summary="获取游戏状态")
def get_game_state(game_id: str, viewer: Optional[str] = None):
    engine = game_manager.get_game(game_id)
    if engine is None:
        raise HTTPException(status_code=404, detail="游戏不存在")
    state = game_manager.get_game_state(game_id, viewer)

    # 附加玩家详细信息
    players = []
    for p in engine.state.players:
        info = {"player_id": p.player_id, "is_alive": p.is_alive, "camp": p.camp.value}
        # 只有游戏结束时才暴露角色
        if engine.state.current_phase.value == "game_over":
            info["role"] = p.role.value
        players.append(info)

    state["players_detail"] = players
    return state


@app.delete("/api/game/{game_id}", summary="删除游戏")
def delete_game(game_id: str):
    if game_manager.delete_game(game_id):
        return {"message": f"游戏 {game_id} 已删除"}
    raise HTTPException(status_code=404, detail="游戏不存在")


# ──────────────────────────── 游戏流程 ────────────────────────────

@app.post("/api/game/{game_id}/step", summary="前进一个阶段（夜晚→讨论→投票）")
def step_game(game_id: str):
    """每次调用前进一个阶段：夜晚 → 白天讨论 → 白天投票 → 夜晚 ... """
    result = game_manager.step_game(game_id)
    if result is None:
        raise HTTPException(status_code=404, detail="游戏不存在")

    state = game_manager.get_game_state(game_id)
    return {"step_result": result, "state": state}


@app.post("/api/game/{game_id}/run", summary="完整运行游戏直到结束")
def run_game(game_id: str):
    """完整运行整局游戏并返回结果"""
    engine = game_manager.get_game(game_id)
    if engine is None:
        raise HTTPException(status_code=404, detail="游戏不存在")

    import threading
    def run_in_background():
        game_manager.run_game(game_id)

    thread = threading.Thread(target=run_in_background, daemon=True)
    thread.start()
    thread.join()

    state = game_manager.get_game_state(game_id)
    return {
        "winner": engine.winner,
        "total_rounds": engine.state.current_round,
        "state": state,
        "logs": engine.game_log,
    }


# ──────────────────────────── 消息 ────────────────────────────

@app.get("/api/game/{game_id}/messages", summary="获取消息记录")
def get_messages(game_id: str, player_id: Optional[str] = None):
    """获取游戏消息。player_id 为空时返回所有消息（管理视角）；指定 player_id 时按角色脱敏。"""
    msgs = game_manager.get_messages(game_id, player_id)
    if msgs is None:
        raise HTTPException(status_code=404, detail="游戏不存在")
    return {"messages": msgs}


@app.get("/api/game/{game_id}/logs", summary="获取游戏日志")
def get_game_logs(game_id: str):
    engine = game_manager.get_game(game_id)
    if engine is None:
        raise HTTPException(status_code=404, detail="游戏不存在")
    return {"logs": engine.game_log}


# ──────────────────────────── 配置 ────────────────────────────

@app.get("/api/config", summary="获取游戏配置")
def get_config():
    return {"config": GAME_CONFIG}


# ──────────────────────────── 前端静态文件 ────────────────────────────

MIME_MAP = {
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".html": "text/html; charset=utf-8",
    ".png": "image/png",
    ".ico": "image/x-icon",
    ".svg": "image/svg+xml",
}


@app.get("/css/{filepath:path}")
def serve_css(filepath: str):
    content = _read_fronted_file(f"css/{filepath}")
    if content is None:
        raise HTTPException(404)
    return Response(content=content, media_type="text/css; charset=utf-8")


@app.get("/js/{filepath:path}")
def serve_js(filepath: str):
    content = _read_fronted_file(f"js/{filepath}")
    if content is None:
        raise HTTPException(404)
    return Response(content=content, media_type="application/javascript; charset=utf-8")


@app.get("/", summary="前端页面")
def serve_frontend():
    content = _read_fronted_file("index.html")
    if content is None:
        return {"message": "前端文件未找到"}
    return Response(content=content, media_type="text/html; charset=utf-8")
