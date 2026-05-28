"""游戏引擎 - 状态机驱动游戏流程"""
import random
from collections import Counter
from typing import Dict, List, Optional

from config.settings import GAME_CONFIG, validate_action
from engine.roles import Role, Camp, Phase, ROLE_CAMP_MAP, ROLE_CN
from engine.game_state import GameState, Player
from hub.message_hub import MessageHub


class GameEngine:
    def __init__(self, config: dict = None):
        self.config = config or GAME_CONFIG
        self.state = GameState()
        self.hub = MessageHub()
        self.agents: Dict[str, object] = {}
        self.game_log: List[str] = []
        self.winner: Optional[str] = None
        self._max_rounds = self.config.get("max_rounds", 10)

    def log(self, msg: str):
        self.game_log.append(msg)
        print(msg)

    def setup(self):
        """初始化游戏：分配角色、创建 Agent"""
        roles_pool = []
        for role_name, count in self.config["roles"].items():
            role = Role(role_name)
            roles_pool.extend([role] * count)
        random.shuffle(roles_pool)

        player_count = self.config["player_count"]
        for i in range(player_count):
            player_id = f"Player_{i + 1}"
            role = roles_pool[i]
            camp = ROLE_CAMP_MAP[role]
            player = Player(player_id=player_id, role=role, camp=camp)
            self.state.add_player(player)

        # 创建 Agent（延迟导入避免循环依赖）
        from agents.mock_agent import create_agent_for_role
        for player in self.state.players:
            agent = create_agent_for_role(player.player_id, player.role)
            self.agents[player.player_id] = agent

        # 通知各 Agent 其角色和阵营
        for pid, agent in self.agents.items():
            player = self.state.get_player_by_id(pid)
            agent.set_role_info(player.role, player.camp)
            # 狼人互相告知队友
            if player.camp == Camp.EVIL:
                teammates = [
                    p.player_id for p in self.state.players
                    if p.camp == Camp.EVIL and p.player_id != pid
                ]
                agent.set_teammates(teammates)

        # 狼人组消息频道
        wolf_ids = [p.player_id for p in self.state.players if p.camp == Camp.EVIL]
        self.hub.set_group_members("werewolf", wolf_ids)

        self._log_setup()

    def _log_setup(self):
        self.log("=" * 50)
        self.log("游戏开始！角色分配：")
        for p in self.state.players:
            self.log(f"  {p.player_id} -> {ROLE_CN[p.role]}（{p.camp.value}）")
        self.log("=" * 50)

    # ────────────────────────────── 分步执行接口 ──────────────────────────────

    def step(self) -> dict:
        """执行当前阶段的一步，返回该阶段的结果"""
        phase = self.state.current_phase

        if phase == Phase.NIGHT:
            return self._step_night()
        elif phase == Phase.DAY_DISCUSS:
            return self._step_day_discuss()
        elif phase == Phase.DAY_VOTE:
            return self._step_day_vote()
        elif phase == Phase.GAME_OVER:
            return {"phase": "game_over", "round": self.state.current_round, "winner": self.winner}

        return {"phase": "unknown", "error": f"未知阶段: {phase}"}

    def _step_night(self) -> dict:
        """执行夜晚阶段"""
        self.state.current_round += 1
        self.log(f"\n{'='*50}")
        self.log(f"  第 {self.state.current_round} 轮")
        self.log(f"{'='*50}")

        self.log("\n--- 夜晚降临 ---")
        self.hub.new_round(self.state.current_round, Phase.NIGHT)

        night_result = {
            "kill_target": None,
            "check_result": {},
            "save_target": None,
            "poison_target": None,
        }

        # 1. 狼人商议击杀
        werewolves = self.state.get_alive_players_by_role(Role.WEREWOLF)
        kill_votes = []
        for wolf in werewolves:
            agent = self.agents[wolf.player_id]
            game_info = self.state.get_snapshot_for_player(wolf.player_id)
            action = agent.decide_night_action(game_info)
            if validate_action(action, "kill"):
                kill_votes.append(action["target"])
                self.log(f"  {wolf.player_id}(狼人) 选择击杀: {action['target']}")
            else:
                self.log(f"  {wolf.player_id}(狼人) 动作无效: {action}")

        if kill_votes:
            counter = Counter(kill_votes)
            night_result["kill_target"] = counter.most_common(1)[0][0]

        # 2. 预言家查验
        seers = self.state.get_alive_players_by_role(Role.SEER)
        for seer in seers:
            agent = self.agents[seer.player_id]
            game_info = self.state.get_snapshot_for_player(seer.player_id)
            action = agent.decide_night_action(game_info)
            if validate_action(action, "check"):
                target = action["target"]
                target_player = self.state.get_player_by_id(target)
                if target_player:
                    result = "狼人" if target_player.camp == Camp.EVIL else "好人"
                    night_result["check_result"][seer.player_id] = {
                        "target": target,
                        "result": result,
                    }
                    self.hub.send_private(
                        "system", seer.player_id,
                        f"查验结果：{target} 的身份是 {result}"
                    )
                    self.log(f"  {seer.player_id}(预言家) 查验 {target} -> {result}")

        # 3. 女巫用药
        witches = self.state.get_alive_players_by_role(Role.WITCH)
        for witch in witches:
            agent = self.agents[witch.player_id]
            game_info = self.state.get_snapshot_for_player(witch.player_id)
            game_info["tonight_kill"] = night_result["kill_target"]
            game_info["antidote_used"] = self.state.witch_antidote_used
            game_info["poison_used"] = self.state.witch_poison_used
            action = agent.decide_night_action(game_info)
            if validate_action(action, "witch_action"):
                if action["save_target"] != "none" and not self.state.witch_antidote_used:
                    night_result["save_target"] = action["save_target"]
                    self.state.witch_antidote_used = True
                    self.log(f"  {witch.player_id}(女巫) 使用解药救助 {action['save_target']}")
                if action["poison_target"] != "none" and not self.state.witch_poison_used:
                    night_result["poison_target"] = action["poison_target"]
                    self.state.witch_poison_used = True
                    self.log(f"  {witch.player_id}(女巫) 使用毒药毒杀 {action['poison_target']}")

        # 4. 结算夜晚结果
        self.state.night_actions[self.state.current_round] = night_result

        killed = []
        kill_target = night_result["kill_target"]
        if kill_target and kill_target != night_result["save_target"]:
            self.state.kill_player(kill_target, cause="werewolf_kill")
            killed.append(kill_target)

        poison_target = night_result["poison_target"]
        if poison_target:
            player = self.state.get_player_by_id(poison_target)
            if player and player.is_alive:
                self.state.kill_player(poison_target, cause="witch_poison")
                killed.append(poison_target)

        night_result["killed"] = killed

        if killed:
            for pid in killed:
                self.hub.broadcast("system", f"昨晚 {pid} 死亡了。")
                self.log(f"  >>> {pid} 在夜晚死亡")
        else:
            self.hub.broadcast("system", "昨晚是平安夜，无人死亡。")
            self.log("  >>> 平安夜，无人死亡")

        # 检查胜负
        win = self.state.check_win_condition()
        if win:
            self._end_game(win)
            self.state.current_phase = Phase.GAME_OVER
        else:
            self.state.current_phase = Phase.DAY_DISCUSS

        return {
            "phase": "night",
            "round": self.state.current_round,
            "winner": win,
            "data": night_result,
        }

    def _step_day_discuss(self) -> dict:
        """执行白天讨论阶段"""
        self.log("\n--- 白天讨论 ---")
        self.hub.new_round(self.state.current_round, Phase.DAY_DISCUSS)

        speeches = []
        alive = self.state.get_alive_players()
        for player in alive:
            agent = self.agents[player.player_id]
            game_info = self.state.get_snapshot_for_player(player.player_id)

            monologue = agent.inner_monologue(game_info)
            if monologue:
                self.log(f"  [{player.player_id} 内心] {monologue}")

            speech = agent.generate_speech(game_info)
            self.hub.broadcast(player.player_id, speech)
            self.log(f"  [{player.player_id}] {speech}")

            speeches.append({
                "player_id": player.player_id,
                "role": player.role.value,
                "monologue": monologue,
                "speech": speech,
            })

        self.state.current_phase = Phase.DAY_VOTE

        return {
            "phase": "day_discuss",
            "round": self.state.current_round,
            "winner": None,
            "data": {"speeches": speeches},
        }

    def _step_day_vote(self) -> dict:
        """执行白天投票阶段"""
        self.log("\n--- 投票阶段 ---")
        self.hub.new_round(self.state.current_round, Phase.DAY_VOTE)

        votes = {}
        alive = self.state.get_alive_players()
        for player in alive:
            agent = self.agents[player.player_id]
            game_info = self.state.get_snapshot_for_player(player.player_id)
            action = agent.decide_vote(game_info)
            if validate_action(action, "vote"):
                votes[player.player_id] = action["target"]
                self.log(f"  {player.player_id} 投票给 {action['target']}（原因: {action['reason']}）")
            else:
                self.log(f"  {player.player_id} 投票无效: {action}")

        self.state.vote_records[self.state.current_round] = votes

        eliminated = None
        vote_counts = Counter([t for t in votes.values() if t != "skip"])
        if vote_counts:
            max_votes = max(vote_counts.values())
            top_voted = [t for t, c in vote_counts.items() if c == max_votes]

            if len(top_voted) == 1:
                eliminated = top_voted[0]
                self.state.kill_player(eliminated, cause="vote")
                self.hub.broadcast("system", f"投票结果：{eliminated} 被投票出局。")
                self.log(f"  >>> {eliminated} 被投票出局（{vote_counts[eliminated]} 票）")
            else:
                self.hub.broadcast("system", "投票平票，无人出局。")
                self.log(f"  >>> 平票，无人出局（{top_voted}）")
        else:
            self.hub.broadcast("system", "无人投票，跳过。")
            self.log("  >>> 无人投票")

        # 检查胜负
        win = self.state.check_win_condition()
        if win:
            self._end_game(win)
            self.state.current_phase = Phase.GAME_OVER
        elif self.state.current_round >= self._max_rounds:
            self._end_game("draw")
            self.state.current_phase = Phase.GAME_OVER
        else:
            self.state.current_phase = Phase.NIGHT

        return {
            "phase": "day_vote",
            "round": self.state.current_round,
            "winner": win,
            "data": {
                "votes": votes,
                "eliminated": eliminated,
            },
        }

    # ────────────────────────────── 原 run() 方法兼容 ──────────────────────────────

    def run(self):
        """主循环（向后兼容，内部调用 step）"""
        self.setup()
        self.state.current_phase = Phase.NIGHT

        while self.state.current_phase != Phase.GAME_OVER:
            self.step()

    # ────────────────────────────── 夜晚阶段（原方法保留） ──────────────────────────────

    def night_phase(self):
        """夜晚阶段（已迁移到 _step_night）"""
        return self._step_night()

    def day_discuss_phase(self):
        """白天讨论阶段（已迁移到 _step_day_discuss）"""
        return self._step_day_discuss()

    def day_vote_phase(self):
        """白天投票阶段（已迁移到 _step_day_vote）"""
        return self._step_day_vote()

    # ────────────────────────────── 游戏结束 ──────────────────────────────

    def _end_game(self, winner: str):
        self.winner = winner
        self.state.current_phase = Phase.GAME_OVER
        self.log(f"\n{'='*50}")
        if winner == "good":
            self.log("游戏结束！好人阵营获胜！")
        elif winner == "evil":
            self.log("游戏结束！狼人阵营获胜！")
        else:
            self.log("游戏结束！平局！")
        self.log(f"{'='*50}")

        self.log("\n最终状态：")
        for p in self.state.players:
            status = "存活" if p.is_alive else "死亡"
            self.log(f"  {p.player_id} - {ROLE_CN[p.role]}（{p.camp.value}）- {status}")
