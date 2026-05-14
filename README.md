# A2A Bridge — 轻量级多 Agent 通信框架

[![Version](https://img.shields.io/badge/version-v2.2.0-blue)](https://github.com/sctale/a2a-bridge/releases)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

> 让两个 AI Agent 通过 HTTP+JSON 互相通信，无需依赖第三方 A2A 库，最小实现，够用就行。

**核心特性：** 事件循环不堵 · 幂等 · 会话亲和 · 异步预热 · httpx 直调 AI

**当前版本：** v2.2.0 — httpx AsyncClient 全局连接池 + 差异化超时 + 错误分类重试

---

## 架构

```
┌─────────────────────────────────────────────────────────┐
│  Agent A (Hermes / Your Agent)                          │
│  ┌─────────────────────────────────────────────────┐   │
│  │  FastAPI Server  :8643                          │   │
│  │  POST /a2a     → AI 对话处理                  │   │
│  │  GET  /health   → 健康检查                      │   │
│  │  GET  /a2a/{id} → 任务状态查询                 │   │
└─────────────────────────────────────────────────────────┘
```
                        │ HTTP POST (JSON)
                        ▼
┌───────────────────────┴─────────────────────────────────┐
│  Agent B (QwenPaw / Claude / Any Agent)                  │
│  ┌─────────────────────────────────────────────────┐   │
│  │  FastAPI Server  :8644                          │   │
│  │  POST /a2a     → AI 对话处理                  │   │
│  │  GET  /health   → 健康检查                      │   │
│  │  GET  /a2a/{id} → 任务状态查询                 │   │
└─────────────────────────────────────────────────────────┘

**关键设计原则：**

1. **不走 ACP/stdio** — Agent 双方在同一网络内，用 HTTP 直连
2. **最小依赖** — 只用 FastAPI + uvicorn + httpx，不绑定任何 Agent 框架
3. **事件循环永远不堵** — 所有同步阻塞代码走 `asyncio.to_thread()`
4. **幂等** — SQLite 缓存，重复请求直接返回，不重复调用 AI
5. **状态机** — `pending → processing → completed/failed`，状态全程可查

---

## 快速开始

### 1. 安装依赖

```bash
pip install fastapi uvicorn httpx
```

### 2. 启动 Agent A（接收方，端口 8643）

```bash
python server_side.py
```

### 3. 启动 Agent B（接收方，端口 8644）

```bash
python client_side.py
```

### 4. 验证连通性
```bash
# A → B：ping
curl -X POST http://localhost:8644/tasks \
  -H "Content-Type: application/json" \
  -d '{
    "version": "1.0",
    "type": "task_delegate",
    "from": "agent-a",
    "to": "agent-b",
    "task_id": "ping-test-001",
    "payload": {
      "task_type": "ping",
      "instruction": ""
    }
  }'

# A → B：对话
curl -X POST http://localhost:8644/tasks \
  -H "Content-Type: application/json" \
  -d '{
    "version": "1.0",
    "type": "task_delegate",
    "from": "agent-a",
    "to": "agent-b",
    "task_id": "chat-test-001",
    "payload": {
      "task_type": "chat",
      "instruction": "你好，介绍你自己"
    }
  }'

# B → A：查询状态
curl http://localhost:8643/a2a/chat-test-001
```

---

## 协议格式

### task_delegate（任务委托）

```json
{
  "version": "1.0",
  "type": "task_delegate",
  "from": "agent-a",
  "to": "agent-b",
  "task_id": "uuid-v4",
  "timestamp": "2026-05-11T20:00:00+08:00",
  "payload": {
    "task_type": "chat | ping | web_search | coding | analysis",
    "instruction": "具体指令",
    "context": {}
  }
}
```

### task_result（任务结果）

```json
{
  "version": "1.0",
  "type": "task_result",
  "from": "agent-b",
  "to": "agent-a",
  "task_id": "uuid-v4",
  "status": "success | failed",
  "result": {
    "output": "AI 响应文本",
    "data": {},
    "attachments": []
  },
  "error": null
}
```

---

## 任务类型（task_type）

| task_type | 说明 | 走 AI | 适用场景 |
|-----------|------|--------|---------|
| `ping` | 存活检测 | ❌ 不走 | 健康检查 |
| `chat` | 简单对话 | ✅ 走 | 轻量闲聊、简单问答 |
| `web_search` | 网络搜索 | ✅ 走 | 查资料、搜财报等 |
| `coding` | 代码开发 | ✅ 走 | 写代码、改 bug |
| `analysis` | 分析报告 | ✅ 走 | 深度分析任务 |

**重要：** `ping` 必须在 AI 初始化之前判断返回，否则冷启动时 ping 也会超时。

```python
# ✅ 正确：ping 判断在 get_agent() 之前
if task_type == "ping":
    return JSONResponse(content=make_response(task_id, "success", result={"output": "pong"}))

# ❌ 错误：get_agent() 在 ping 判断之前，AI 冷启动堵住所有请求
agent = get_agent()
if task_type == "ping":
    return JSONResponse(...)
```

---

## 核心实现细节

### 1. 事件循环不堵

**原则：** FastAPI `async def` 里所有同步阻塞代码必须进线程池。

```python
import asyncio

async def handle_request(request):
    # ✅ 正确：同步代码走线程池，事件循环始终保持响应
    result = await asyncio.to_thread(some_sync_function, arg1, arg2)

    # ❌ 错误：同步阻塞，事件循环卡死，同时只能处理一个请求
    result = some_sync_function(arg1, arg2)
```

### 2. AI 异步预热

**问题：** 首次 AI 初始化冷启动 10s+，所有请求（包括 ping）被堵死。

**解决：** 启动时异步预热，不阻塞请求处理。

```python
_agent = None

@app.on_event("startup")
async def startup():
    asyncio.create_task(preload_agent())

async def preload_agent():
    global _agent
    _agent = await asyncio.to_thread(load_agent_sync)  # 后台加载
```

### 3. 幂等存储

**原理：** SQLite + 文件锁，`task_id` 查重，命中直接返回缓存。

```python
class IdempotencyStore:
    def __init__(self, db_path="/tmp/a2a_idempotency.db"):
        self._lock = threading.Lock()
        self._db = sqlite3.connect(db_path, timeout=5, check_same_thread=False)
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS processed_tasks (
                task_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                result TEXT,
                completed_at TEXT NOT NULL
            )
        """)

    def get(self, task_id):
        with self._lock:
            row = self._db.execute(
                "SELECT status, result FROM processed_tasks WHERE task_id = ?",
                (task_id,)
            ).fetchone()
            return {"status": row[0], "result": json.loads(row[1])} if row else None

    def set(self, task_id, status, result):
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO processed_tasks VALUES (?, ?, ?, ?)",
                (task_id, status, json.dumps(result), datetime.now().isoformat())
            )
            self._db.commit()
```

### 4. 状态机

```
pending ──→ processing ──→ completed
                    │
                    └──→ failed
```

每个任务存储：`status` + `created_at` + `started_at` + `completed_at`

---

## Docker 部署

同一主机不同容器或不同 Docker Compose 项目间通信，需要端口映射。

```yaml
# docker-compose.yml（Agent A）
services:
  agent-a:
    ports:
      - "8643:8643"
    # ...

# docker-compose.yml（Agent B）
services:
  agent-b:
    ports:
      - "8644:8644"
    # ...
```

通信地址：
- A → B：`http://<host>:8644/a2a`
- B → A：`http://<host>:8643/tasks`

---

## 常见陷阱

### 1. API Key 读取

**问题：** 独立进程不继承父进程环境变量，`os.environ.get()` 可能拿到空值。

**解决：** 直接读 `.env` 文件解析。

```python
def load_api_key(env_path="/data/.env"):
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                if "API_KEY" in line and not line.strip().startswith("#"):
                    return line.split("=", 1)[1].strip()
    return ""
```

### 2. task_type 位置

**问题：** `task_type` 可能藏在 `context` 里而不是 `payload` 里。

**解决：** 同时检查两个位置。

```python
task_type = context.get("task_type") or payload.get("task_type", "chat")
```

### 3. A2A 消息不能传代码

**问题：** JSON 消息体传代码会截断/编码异常。

**解决：** A2A 消息只发路径和改动说明，代码写文件传递。

---

## 目录结构
```
.
├── README.md              # 本文件
├── LICENSE               # MIT License
├── requirements.txt      # 依赖
├── server/
│   ├── __init__.py
│   └── server_side.py   # Agent A（A2A Server，端口 8643）
├── client/
│   ├── __init__.py
│   └── client_side.py   # Agent B（A2A Server，端口 8644）
└── examples/
    └── docker-compose/  # Docker 部署示例
```

---

## 版本历史
## 版本历史
| 版本 | 日期 | 变更 |
|------|------|------|
| v1.0.0 | 2026-05-12 | 初始版本（subprocess 方式） |
| v2.0.0 | 2026-05-13 | httpx AsyncClient 直调 AI，替代 subprocess；完整幂等存储；Agent A + Agent B 双端 |
| **v2.1.0** | 2026-05-13 | 会话亲和（session_id 相同则自动注入历史上下文，支持多轮对话） |
| **v2.2.0** | 2026-05-13 | httpx 全局连接池 + 差异化超时（ping 5s/chat 15s/search 60s）+ 错误分类重试（429 指数退避 / 500 等1s / 4xx 不重试）+ 幂等 TTL 24h |

## 扩展方向

- [x] 健康检查探针（readiness / liveness）
- [x] httpx AsyncClient 直调 AI（替代 subprocess，解决冷启动）— v2.0.0
- [x] 会话亲和（session_id 相同则自动注入历史上下文）— v2.1.0
- [x] httpx 全局连接池 + 差异化超时 + 错误分类重试 — v2.2.0
- [ ] 支持 WebSocket 双向推送
- [ ] 消息持久化（SQLite 表，支持重启恢复）
- [ ] mTLS 双向认证
- [ ] 多 Agent 路由（N > 2）

---

## httpx 直调 AI 方案（v2.0.0+）

### 核心思路

Agent 在收到 A2A 请求时，直接用 `httpx.AsyncClient` 调 AI Provider API，不 fork 子进程，不阻塞事件循环。

```python
# httpx 全局客户端（连接池复用）
_httpx_client: httpx.AsyncClient = None

def get_httpx_client() -> httpx.AsyncClient:
    global _httpx_client
    if _httpx_client is None or _httpx_client.is_closed:
        _httpx_client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=5.0, read=30.0, write=10.0, pool=5.0),
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
        )
    return _httpx_client

# AI 调用（异步 httpx，事件循环不堵）
async def call_ai(system_prompt, instruction, session_id=None, task_type="chat") -> str:
    client = get_httpx_client()
    resp = await client.post(
        f"{base_url}/v1/messages",
        headers={"Authorization": f"Bearer {api_key}", ...},
        json={"model": model, "messages": [...], "system": system_prompt},
        timeout=_TASK_TIMEOUTS.get(task_type, 15.0),
    )
    resp.raise_for_status()
    data = resp.json()
    return data["content"][0]["text"]
```

### 关键设计

- **ping 提前返回**：`task_type == "ping"` 在任何 AI 调用之前判断，不走 httpx
- **差异化超时**：ping 5s / chat 15s / search 60s
- **错误分类重试**：429 指数退避 / 500 等1s / 4xx 直接抛
- **连接池复用**：`max_connections=100, max_keepalive=20`，高并发不重建连接


