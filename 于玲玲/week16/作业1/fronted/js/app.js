const API_BASE = '/api';

const { createApp, ref, computed, nextTick, watch } = Vue;

createApp({
  setup() {
    // ========== 响应式状态 ==========
    const gameId = ref(null);
    const gameState = ref(null);
    const players = ref([]);
    const messages = ref([]);
    const logs = ref([]);
    const stepResult = ref(null);
    const gameOver = ref(false);
    const winner = ref(null);
    const loading = ref(false);
    const error = ref('');
    const serverConnected = ref(false);
    const activeTab = ref('messages');
    const messagesRef = ref(null);
    const logsRef = ref(null);
    const createConfig = ref({
      player_count: 5,
      max_rounds: 10,
    });

    // ========== 计算属性 ==========
    const phaseLabel = computed(() => {
      if (!gameState.value) return '未开始';
      const map = {
        'night': '🌙 夜晚',
        'day_discuss': '☀️ 白天讨论',
        'day_vote': '🗳️ 白天投票',
        'game_over': '🏁 游戏结束',
      };
      return map[gameState.value.phase] || gameState.value.phase;
    });

    const phaseClass = computed(() => {
      if (!gameState.value) return '';
      return 'phase-' + gameState.value.phase;
    });

    const winnerClass = computed(() => {
      if (winner.value === 'good') return 'winner-good';
      if (winner.value === 'evil') return 'winner-evil';
      return 'winner-draw';
    });

    const winnerText = computed(() => {
      if (winner.value === 'good') return '👼 好人阵营获胜！';
      if (winner.value === 'evil') return '🐺 狼人阵营获胜！';
      return '🤝 平局！';
    });

    // ========== 工具函数 ==========
    function roleCN(role) {
      const map = {
        werewolf: '狼人',
        seer: '预言家',
        witch: '女巫',
        villager: '村民',
      };
      return map[role] || role;
    }

    function campCN(camp) {
      return camp === 'evil' ? '狼人阵营' : '好人阵营';
    }

    function logLineClass(line) {
      if (line.includes('>>>') || line.includes('死亡')) return 'log-death';
      if (line.includes('投票')) return 'log-vote';
      if (line.includes('内心') || line.includes('发言')) return 'log-discuss';
      if (line.includes('夜晚') || line.includes('狼人') || line.includes('查验') || line.includes('解药') || line.includes('毒药')) return 'log-night';
      if (line.includes('第') && line.includes('轮') && !line.includes('---')) return 'log-round';
      return '';
    }

    // ========== API 通用请求 ==========
    async function apiFetch(url, options = {}) {
      try {
        const resp = await fetch(API_BASE + url, {
          headers: { 'Content-Type': 'application/json' },
          ...options,
        });
        if (!resp.ok) {
          const detail = await resp.json().catch(() => ({ detail: `HTTP ${resp.status}` }));
          throw new Error(detail.detail || `请求失败 (${resp.status})`);
        }
        return await resp.json();
      } catch (err) {
        if (err.message.includes('Failed to fetch') || err.message.includes('NetworkError')) {
          serverConnected.value = false;
          throw new Error('无法连接到服务器，请确认服务已启动');
        }
        throw err;
      }
    }

    // ========== API 方法 ==========

    // 检测服务器连接
    async function checkServer() {
      try {
        await apiFetch('/config');
        serverConnected.value = true;
      } catch {
        serverConnected.value = false;
      }
    }

    // 创建游戏
    async function createGame() {
      error.value = '';
      loading.value = true;
      gameOver.value = false;
      stepResult.value = null;

      try {
        const data = await apiFetch('/game', {
          method: 'POST',
          body: JSON.stringify({
            player_count: createConfig.value.player_count,
            max_rounds: createConfig.value.max_rounds,
          }),
        });
        gameId.value = data.game_id;
        gameState.value = data.state;
        serverConnected.value = true;
        await refreshPlayers();
        await refreshMessages();
        await refreshLogs();
      } catch (err) {
        error.value = err.message;
      } finally {
        loading.value = false;
      }
    }

    // 单步执行
    async function stepGame() {
      if (!gameId.value) return;
      error.value = '';
      loading.value = true;

      try {
        const data = await apiFetch(`/game/${gameId.value}/step`, {
          method: 'POST',
        });
        stepResult.value = data.step_result;
        gameState.value = data.state;

        // 检查游戏是否结束
        if (data.step_result.phase === 'game_over' || data.state?.phase === 'game_over') {
          winner.value = data.step_result.winner || data.state?.winner;
          gameOver.value = true;
        }

        serverConnected.value = true;
        await refreshPlayers();
        await refreshMessages();
        await refreshLogs();
      } catch (err) {
        error.value = err.message;
      } finally {
        loading.value = false;
      }
    }

    // 完整运行
    async function runGame() {
      if (!gameId.value) return;
      error.value = '';
      loading.value = true;

      try {
        const data = await apiFetch(`/game/${gameId.value}/run`, {
          method: 'POST',
        });
        winner.value = data.winner;
        gameOver.value = true;
        stepResult.value = { phase: 'game_over', winner: data.winner, total_rounds: data.total_rounds };
        gameState.value = data.state;
        logs.value = data.logs || [];
        serverConnected.value = true;
        await refreshPlayers();
        await refreshMessages();
      } catch (err) {
        error.value = err.message;
      } finally {
        loading.value = false;
      }
    }

    // 刷新玩家列表
    async function refreshPlayers() {
      if (!gameId.value) return;
      try {
        const data = await apiFetch(`/game/${gameId.value}`);
        players.value = data.players_detail || [];
      } catch {
        // 静默失败
      }
    }

    // 刷新消息
    async function refreshMessages() {
      if (!gameId.value) return;
      try {
        const data = await apiFetch(`/game/${gameId.value}/messages`);
        messages.value = data.messages || [];
        await nextTick();
        // 滚动到底部
        if (messagesRef.value) {
          messagesRef.value.scrollTop = messagesRef.value.scrollHeight;
        }
      } catch {
        // 静默失败
      }
    }

    // 刷新日志
    async function refreshLogs() {
      if (!gameId.value) return;
      try {
        const data = await apiFetch(`/game/${gameId.value}/logs`);
        logs.value = data.logs || [];
        await nextTick();
        // 滚动到底部
        if (logsRef.value) {
          logsRef.value.scrollTop = logsRef.value.scrollHeight;
        }
      } catch {
        // 静默失败
      }
    }

    // 离开当前游戏
    function leaveGame() {
      gameId.value = null;
      gameState.value = null;
      players.value = [];
      messages.value = [];
      logs.value = [];
      stepResult.value = null;
      gameOver.value = false;
      winner.value = null;
      error.value = '';
      loading.value = false;
      activeTab.value = 'messages';
    }

    // ========== 初始化 ==========
    checkServer();

    return {
      // 状态
      gameId, gameState, players, messages, logs, stepResult,
      gameOver, winner, loading, error, serverConnected, activeTab,
      createConfig, messagesRef, logsRef,
      // 计算属性
      phaseLabel, phaseClass, winnerClass, winnerText,
      // 方法
      createGame, stepGame, runGame, leaveGame,
      roleCN, campCN, logLineClass,
    };
  },
}).mount('#app');
