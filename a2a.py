#!/usr/bin/env python3
"""
A2A Bridge — 通用双向 Agent 通信节点 v1.2.0

设计原则：
- 每个实例既是 Server（接收任务）也是 Client（发送任务）
- 不分客户端/服务端，只有 port + peer 配置
- 环境变量驱动，所有配置通过 env 注入
- 无需 API Key，内网互通

启动：
    A2A_PORT=8643 A2A_PEER_URL=http://localhost:8644 python a2a.py

环境变量：
    A2A_PORT              监听端口（默认 8643）
    A2A_PEER_URL          对端地址，用于主动发任务（默认空，不主动发送）
    A2A_NAME              节点名称（默认 HOSTNAME 或 "node-{port}"）
    AI_PROVIDER_BASE_URL  AI API 地址（默认 https://api.minimaxi.com/anthropic）
    A2A_MODEL             AI 模型（默认 gpt-4o）
    A2A_MAX_TOKENS        最大 token 数（默认 2048）
    A2A_IDEMPOTENCY_DB    幂等 DB 路径（默认 /tmp/a2a_idempotency_{port}.db）
    A2A_SESSION_DB        会话 DB 路径（默认 /tmp/a2a_sessions_{port}.db）
    A2A_ENV_PATH          .env 文件路径（用于 docker 挂载 .env）
"""
import asyncio
import datetime as dt
import json
import logging
import os
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
import uvicorn

__version__ = "1.2.0"

# ─── 日志 ─────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("a2a")

# ─── 配置读取 ─────────────────────────────────────────────────────────

def _load_env(key: str) -> str:
    """从环境变量读取，失败则尝试 .env 文件（支持 docker 挂载）"""
    val = os.environ.get(key, "")
    if val:
        return val
    env_path = os.environ.get("A2A_ENV_PATH", ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                ls = line.strip()
                if ls.startswith("#") or "=" not in ls:
                    continue
                k, v = ls.split("=", 1)
                if k.strip() == key:
                    return v.strip()
    return ""


# 节点自身配置
MY_PORT = int(os.environ.get("A2A_PORT", "8643"))
MY_NAME = os.environ.get("A2A_NAME", "") or os.environ.get("HOSTNAME", f"node-{MY_PORT}")
MY_PEER_URL = os.environ.get("A2A_PEER_URL", "").rstrip("/")  # 空=仅接收，不主动发送

# AI 配置
AI_BASE_URL = _load_env("AI_PROVIDER_BASE_URL") or "https://api.minimaxi.com/anthropic"
AI_MODEL = os.environ.get("A2A_MODEL", "gpt-4o")
AI_API_KEY = _load_env("AI_PROVIDER_API_KEY")  # 从 .env 读取，不再从请求 header
AI_MAX_TOKENS = int(os.environ.get("A2A_MAX_TOKENS", "2048"))

# 数据库路径（每个端口独立 DB，重启不混淆）
IDEMPOTENCY_DB = os.environ.get("A2A_IDEMPOTENCY_DB", f"/tmp/a2a_idempotency_{MY_PORT}.db")
SESSION_DB = os.environ.get("A2A_SESSION_DB", f"/tmp/a2a_sessions_{MY_PORT}.db")

logger.info(f"[{MY_NAME}] 启动 A2A Bridge，port={MY_PORT} peer={MY_PEER_URL or '(仅接收)'}")
if not AI_API_KEY:
    logger.warning(f"[{MY_NAME}] AI_PROVIDER_API_KEY 未配置，AI 调用可能失败")

# ─── FastAPI App ──────────────────────────────────────────────────────

app = FastAPI(title=f"A2A Bridge — {MY_NAME}", version=__version__, description="通用双向 Agent 通信节点")

# 内存任务存储（状态查询用，不做持久化）
tasks: Dict[str, Dict[str, Any]] = {}

# ─── httpx 全局客户端（连接池复用）───────────────────────────────────

_httpx_client: Optional[httpx.AsyncClient] = None


def get_httpx_client() -> httpx.AsyncClient:
    global _httpx_client
    if _httpx_client is None or _httpx_client.is_closed:
        _httpx_client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=5.0, read=60.0, write=10.0, pool=5.0),
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
        )
    return _httpx_client


# ─── 幂等存储（SQLite，TTL 24h，failed 可重试）────────────────────────

class IdempotencyStore:
    TTL_HOURS = 24

    def __init__(self, db_path: str):
        self._lock = threading.Lock()
        self._db = sqlite3.connect(db_path, timeout=5, check_same_thread=False)
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS processed_tasks (
                task_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                result TEXT,
                created_at TEXT NOT NULL,
                completed_at TEXT NOT NULL
            )
        """)
        self._db.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON processed_tasks(created_at)")

    def get(self, task_id: str) -> Optional[Dict]:
        with self._lock:
            row = self._db.execute(
                "SELECT status, result, created_at FROM processed_tasks WHERE task_id = ?",
                (task_id,),
            ).fetchone()
            if not row:
                return None
            status, result_json, created_at = row
            # failed 允许重试，TTL 过期也允许重试
            if status == "failed":
                self._db.execute("DELETE FROM processed_tasks WHERE task_id = ?", (task_id,))
                self._db.commit()
                return None
            created = dt.datetime.fromisoformat(created_at)
            if (dt.datetime.now() - created).total_seconds() > self.TTL_HOURS * 3600:
                self._db.execute("DELETE FROM processed_tasks WHERE task_id = ?", (task_id,))
                self._db.commit()
                return None
            return {
                "status": status,
                "result": json.loads(result_json) if result_json else None,
            }

    def set(self, task_id: str, status: str, result: Optional[Dict] = None):
        now = datetime.now().isoformat()
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO processed_tasks (task_id, status, result, created_at, completed_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (task_id, status, json.dumps(result, ensure_ascii=False), now, now),
            )
            self._db.commit()


# ─── 会话亲和存储（SQLite，TTL + 阈值压缩）─────────────────────────────

class SessionStore:
    COMPRESS_THRESHOLD = 12
    KEEP_RECENT = 3
    TTL_DAYS = 7

    def __init__(self, db_path: str):
        self._lock = threading.Lock()
        self._db = sqlite3.connect(db_path, timeout=5, check_same_thread=False)
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS a2a_sessions (
                session_id TEXT PRIMARY KEY,
                messages TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        self._db.execute("CREATE INDEX IF NOT EXISTS idx_updated_at ON a2a_sessions(updated_at)")

    def get_history(self, session_id: str, limit: int = 10) -> list:
        with self._lock:
            row = self._db.execute(
                "SELECT messages FROM a2a_sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
            if not row:
                return []
            msgs = json.loads(row[0])
            return msgs[-limit:]

    def append(self, session_id: str, role: str, content: str, limit: int = 20):
        with self._lock:
            row = self._db.execute(
                "SELECT messages FROM a2a_sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
            msgs = json.loads(row[0]) if row else []
            msgs.append({"role": role, "content": content})
            if len(msgs) > self.COMPRESS_THRESHOLD:
                msgs = self._compress(msgs)
            msgs = msgs[-limit:]
            self._db.execute(
                "INSERT OR REPLACE INTO a2a_sessions (session_id, messages, updated_at) VALUES (?, ?, ?)",
                (session_id, json.dumps(msgs, ensure_ascii=False), datetime.now().isoformat()),
            )
            self._db.commit()

    def _compress(self, msgs: list) -> list:
        parts = []
        for m in msgs[: -self.KEEP_RECENT]:
            if m.get("role") in ("user", "assistant"):
                c = m["content"]
                if len(c) > 60:
                    c = c[:60] + "..."
                parts.append(f"{m['role']}: {c}")
        summary = f"[上文摘要({len(msgs)}条): {'; '.join(parts[:8])}]"
        system_msgs = [m for m in msgs if m.get("role") == "system"]
        return system_msgs + [{"role": "system", "content": summary}] + msgs[-self.KEEP_RECENT:]

    def cleanup_expired(self):
        cutoff = (dt.datetime.now() - dt.timedelta(days=self.TTL_DAYS)).isoformat()
        with self._lock:
            self._db.execute("DELETE FROM a2a_sessions WHERE updated_at < ?", (cutoff,))
            self._db.commit()


_idempotency = IdempotencyStore(IDEMPOTENCY_DB)
_session_store = SessionStore(SESSION_DB)

# ─── 超时配置 ─────────────────────────────────────────────────────────

_TASK_TIMEOUTS: Dict[str, float] = {
    "ping": 5.0,
    "chat": 120.0,
    "web_search": 120.0,
    "coding": 120.0,
    "analysis": 120.0,
}

# ─── AI 调用 ──────────────────────────────────────────────────────────


async def call_ai(
    system_prompt: str,
    instruction: str,
    session_id: Optional[str] = None,
    task_type: str = "chat",
) -> str:
    """异步 httpx 调用 AI，API Key 从环境变量读取"""
    if not AI_API_KEY:
        raise ValueError("AI_PROVIDER_API_KEY 未配置")

    timeout = _TASK_TIMEOUTS.get(task_type, 120.0)
    messages = _build_messages(instruction, session_id)
    payload = {
        "model": AI_MODEL,
        "max_tokens": AI_MAX_TOKENS,
        "system": system_prompt,
        "messages": messages,
    }

    last_exc = None
    for attempt in range(3):
        try:
            client = get_httpx_client()
            resp = await client.post(
                f"{AI_BASE_URL}/v1/messages",
                headers={
                    "Authorization": f"Bearer {AI_API_KEY}",
                    "Content-Type": "application/json",
                    "x-api-key": AI_API_KEY,
                    "anthropic-version": "2023-06-01",
                },
                json=payload,
                timeout=timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            for item in data.get("content", []):
                if item.get("type") == "text":
                    text = item.get("text", "")
                    # 存储会话历史
                    if session_id:
                        _session_store.append(session_id, "user", instruction)
                        _session_store.append(session_id, "assistant", text)
                    return text
            return str(data.get("content", []))

        except httpx.HTTPStatusError as exc:
            last_exc = exc
            if exc.response.status_code == 429:
                wait = 2 ** attempt
                logger.warning(f"[{MY_NAME}] 429 限流，等待 {wait}s 重试")
                await asyncio.sleep(wait)
                continue
            if exc.response.status_code >= 500:
                logger.warning(f"[{MY_NAME}] 服务端错误 {exc.response.status_code}，等待 1s 重试")
                await asyncio.sleep(1)
                continue
            raise

        except (httpx.ReadTimeout, httpx.ConnectError) as exc:
            last_exc = exc
            if attempt < 2:
                logger.warning(f"[{MY_NAME}] 网络异常 {type(exc).__name__}，等待 1s 重试")
                await asyncio.sleep(1)
                continue
            raise

    raise last_exc or RuntimeError("AI 调用失败")


def _build_messages(instruction: str, session_id: Optional[str] = None) -> list:
    messages = []
    if session_id:
        messages.extend(_session_store.get_history(session_id, limit=10))
    messages.append({"role": "user", "content": instruction})
    return messages


# ─── 工具函数 ─────────────────────────────────────────────────────────

def make_response(
    task_id: str,
    status: str,
    result: Optional[Dict] = None,
    error: Optional[Dict] = None,
    from_agent: str = MY_NAME,
    to_agent: str = "*",
) -> Dict:
    return {
        "version": "1.0",
        "type": "task_result",
        "from": from_agent,
        "to": to_agent,
        "task_id": task_id,
        "status": status,
        "result": result,
        "error": error,
    }


def _resolve_task_type(payload: Dict, context: Dict) -> str:
    return context.get("task_type") or payload.get("task_type", "chat")


def _get_local_ip() -> str:
    """获取本机局域网 IP（用于 reply_to 地址）"""
    try:
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("192.168.20.252", 8644))  # 连接对端获取路由出口IP
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


# ─── HTTP 端点 ────────────────────────────────────────────────────────

@app.get("/health")
async def health(request: Request):
    scheme = request.headers.get("x-forwarded-proto", "http")
    host = request.headers.get("host", f"localhost:{MY_PORT}")
    agent_card_url = f"{scheme}://{host}/.well-known/agent-card.json"
    return {
        "status": "ok",
        "name": MY_NAME,
        "port": MY_PORT,
        "peer": MY_PEER_URL or None,
        "version": __version__,
        "agentCardUrl": agent_card_url,
    }


@app.get("/ready")
async def ready():
    return {"status": "ready", "name": MY_NAME}


@app.get("/live")
async def live():
    return {"status": "ok"}


@app.get("/capabilities")
async def capabilities():
    return {
        "version": __version__,
        "name": MY_NAME,
        "capabilities": {
            "streaming": False,
            "pushNotifications": False,
            "extendedAgentCard": False,
        },
        "skills": [
            {"id": "chat", "name": "AI 对话", "description": "通用 AI 对话助手，回答简洁直接"},
            {"id": "web_search", "name": "网络搜索", "description": "互联网信息搜索与摘要"},
            {"id": "coding", "name": "代码开发", "description": "代码编写、调试与优化"},
            {"id": "analysis", "name": "分析报告", "description": "数据与文本深度分析"},
        ],
        "features": {
            "idempotency": True,
            "session_affinity": True,
            "async_preload": True,
            "capabilities_endpoint": True,
            "version_negotiation": True,
        },
    }


def _build_agent_card(request: Request) -> Dict:
    """构建标准 Agent Card JSON"""
    scheme = request.headers.get("x-forwarded-proto", "http")
    host = request.headers.get("host", f"localhost:{MY_PORT}")
    base_url = f"{scheme}://{host}"
    return {
        "name": MY_NAME,
        "description": f"A2A Bridge Agent — {MY_NAME}，支持 ping/chat/web_search/coding/analysis 任务类型",
        "url": base_url,
        "version": __version__,
        "capabilities": {
            "streaming": False,
            "pushNotifications": False,
            "extendedAgentCard": False,
        },
        "skills": [
            {"id": "chat", "name": "AI 对话", "description": "通用 AI 对话助手，回答简洁直接"},
            {"id": "web_search", "name": "网络搜索", "description": "互联网信息搜索与摘要"},
            {"id": "coding", "name": "代码开发", "description": "代码编写、调试与优化"},
            {"id": "analysis", "name": "分析报告", "description": "数据与文本深度分析"},
        ],
    }


@app.get("/.well-known/agent-card.json")
async def agent_card(request: Request):
    """官方 A2A Protocol Agent Card 端点"""
    return JSONResponse(
        content=_build_agent_card(request),
        headers={"Cache-Control": "public, max-age=300"},
    )


@app.get("/a2a/.well-known/agent-card.json")
async def agent_card_alias(request: Request):
    """A2A 官方路径别名（兼容官方客户端）"""
    return JSONResponse(
        content=_build_agent_card(request),
        headers={"Cache-Control": "public, max-age=300"},
    )


@app.post("/a2a")
async def receive_task(request: Request):
    """接收来自对端的任务"""
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "Invalid JSON body"})

    task_id = body.get("task_id") or str(uuid.uuid4())
    from_agent = body.get("from", "unknown")
    to_agent = body.get("to", "*")
    msg_type = body.get("type")
    payload = body.get("payload", {})
    context = payload.get("context", {})
    session_id = payload.get("session_id") or context.get("session_id")

    logger.info(f"[{MY_NAME}] ← 任务 {task_id} from {from_agent} → {to_agent} ({msg_type})")

    # ── task_result 中继：收到对端结果，转发给目标 ─────────────────────
    if msg_type == "task_result":
        # 有 to_agent 且不是自己，说明是relay，直接转发
        if to_agent and to_agent not in (MY_NAME, "*", ""):
            if MY_PEER_URL:
                asyncio.create_task(_relay_result(body))
            else:
                logger.warning(f"[{MY_NAME}] task_result relay 需要 A2A_PEER_URL，当前未配置")
        return JSONResponse(content=make_response(
            task_id=task_id, status="received", to_agent=from_agent,
        ))

    if msg_type != "task_delegate":
        return JSONResponse(status_code=400, content=make_response(
            task_id=task_id, status="failed",
            error={"code": 400, "message": f"不支持的消息类型: {msg_type}"},
            to_agent=from_agent,
        ))

    if to_agent not in (MY_NAME, "*", ""):
        return JSONResponse(status_code=400, content=make_response(
            task_id=task_id, status="failed",
            error={"code": 400, "message": f"未知目标: {to_agent}"},
            to_agent=from_agent,
        ))

    instruction = payload.get("instruction", "")
    task_type = _resolve_task_type(payload, context)

    if not instruction and task_type != "ping":
        return JSONResponse(status_code=400, content=make_response(
            task_id=task_id, status="failed",
            error={"code": 400, "message": "payload.instruction 不能为空"},
            to_agent=from_agent,
        ))

    # ── 幂等检查 ──
    cached = _idempotency.get(task_id)
    if cached:
        logger.info(f"[{MY_NAME}] 幂等命中 {task_id}")
        return JSONResponse(content=make_response(
            task_id=task_id,
            status=cached["status"],
            result=cached["result"],
            to_agent=from_agent,
        ))

    tasks[task_id] = {
        "status": "pending",
        "created_at": datetime.now().isoformat(),
        "from": from_agent,
        "task_type": task_type,
        "instruction": instruction,
    }

    # ── ping 直接返回 ──
    if task_type == "ping":
        tasks[task_id]["status"] = "completed"
        result = {"output": "pong", "data": {"task_id": task_id}, "attachments": []}
        _idempotency.set(task_id, "completed", result)
        return JSONResponse(content=make_response(
            task_id=task_id, status="success", result=result, to_agent=from_agent,
        ))

    # ── AI 调用 ──
    system_prompt = (
        f"你是 {MY_NAME}，一个 AI 助手。回答简洁直接，一针见血。\n"
        "遇到问题先想清楚再回答，不要瞎猜。\n"
        "不知道就说不知道，不编数据。"
    )
    if context:
        system_prompt += f"\n\n附加上下文: {json.dumps(context, ensure_ascii=False)}"

    tasks[task_id]["status"] = "processing"
    try:
        response_text = await call_ai(
            system_prompt, instruction, session_id, task_type,
        )
    except Exception as exc:
        logger.exception(f"[{MY_NAME}] AI 调用异常: {exc}")
        response_text = f"(AI 处理异常: {exc})"

    logger.info(f"[{MY_NAME}] AI 响应: {response_text[:80]}...")

    result = {"output": response_text, "data": {"task_id": task_id}, "attachments": []}
    tasks[task_id]["status"] = "completed"
    _idempotency.set(task_id, "success", result)

    return JSONResponse(content=make_response(
        task_id=task_id, status="success", result=result, to_agent=from_agent,
    ))


async def _relay_result(body: Dict):
    """将 task_result 异步转发给对端"""
    try:
        client = get_httpx_client()
        resp = await client.post(
            f"{MY_PEER_URL}/a2a",
            json=body,
            headers={"Content-Type": "application/json"},
            timeout=30.0,
        )
        resp.raise_for_status()
        logger.info(f"[{MY_NAME}] task_result relay 成功")
    except Exception as exc:
        logger.warning(f"[{MY_NAME}] task_result relay 失败: {exc}")


@app.post("/report")
async def receive_report(request: Request):
    """接收对端的工作汇报，回写到原始任务的 idempotency 缓存"""
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "Invalid JSON body"})

    task_id = body.get("task_id") or str(uuid.uuid4())
    from_agent = body.get("from", "unknown")
    result = body.get("result")
    status = body.get("status", "success")

    logger.info(f"[{MY_NAME}] ← 汇报 from {from_agent}: task_id={task_id} status={status}")

    # 回写结果到幂等缓存，唤醒等待中的同步调用
    if result:
        _idempotency.set(task_id, status, result)

    # 返回确认
    return JSONResponse(content=make_response(
        task_id=task_id,
        status="received",
        to_agent=from_agent,
    ))


@app.get("/a2a/{task_id}")
async def get_task(task_id: str):
    """查询任务状态（优先查 SQLite 持久化，备用内存）"""
    if task_id in tasks:
        return tasks[task_id]
    cached = _idempotency.get(task_id)
    if cached:
        return {
            "task_id": task_id,
            "status": cached["status"],
            "result": cached["result"],
        }
    raise HTTPException(status_code=404, detail=f"任务 {task_id} 不存在")


@app.post("/send")
async def send_task(request: Request):
    """主动向对端发送任务（需配置 A2A_PEER_URL）"""
    if not MY_PEER_URL:
        return JSONResponse(status_code=400, content=make_response(
            task_id="", status="failed",
            error={"code": 400, "message": "A2A_PEER_URL 未配置，无法主动发送任务"},
        ))

    try:
        body = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content=make_response(
            task_id="", status="failed",
            error={"code": 400, "message": "Invalid JSON body"},
        ))

    task_id = body.get("task_id") or str(uuid.uuid4())
    task_type = body.get("task_type", "chat")
    instruction = body.get("instruction", "")
    context = body.get("context", {})
    session_id = body.get("session_id")

    if not instruction and task_type != "ping":
        return JSONResponse(status_code=400, content=make_response(
            task_id=task_id, status="failed",
            error={"code": 400, "message": "instruction 不能为空"},
        ))

    logger.info(f"[{MY_NAME}] → 发送任务 {task_id} to {MY_PEER_URL} ({task_type})")

    payload = {
        "version": "1.0",
        "type": "task_delegate",
        "from": MY_NAME,
        "to": "*",
        "task_id": task_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "payload": {
            "task_type": task_type,
            "instruction": instruction,
            "context": context,
            "session_id": session_id,
            "reply_to": f"http://{_get_local_ip()}:{MY_PORT}",
        },
    }

    cached = _idempotency.get(task_id)
    if cached:
        logger.info(f"[{MY_NAME}] 主动发送幂等命中 {task_id}")
        return JSONResponse(content=make_response(
            task_id=task_id, status=cached["status"], result=cached["result"],
        ))

    # sync=true 时同步等待结果（默认行为，保留兼容）
    # sync=false 时立即返回 queued（并发场景）
    sync = body.get("sync", True)
    timeout = _TASK_TIMEOUTS.get(task_type, 120.0)
    client = get_httpx_client()

    if sync:
        # ── 同步模式：等待对端处理完返回结果 ─────────────────────────────
        try:
            resp = await client.post(
                f"{MY_PEER_URL}/a2a",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=timeout,
            )
            resp.raise_for_status()
            result = resp.json()
        except Exception as exc:
            logger.exception(f"[{MY_NAME}] 任务 {task_id} 失败: {exc}")
            return JSONResponse(status_code=500, content=make_response(
                task_id=task_id, status="failed",
                error={"code": 500, "message": str(exc)},
            ))

        if result.get("status") == "success":
            _idempotency.set(task_id, "success", result.get("result"))

        return JSONResponse(content=make_response(
            task_id=task_id,
            status=result.get("status", "success"),
            result=result.get("result"),
            to_agent=MY_NAME,
        ))

    # ── 并发模式：立即返回 queued，结果通过 /a2a/{task_id} 查询 ─────────
    asyncio.create_task(_send_and_poll(task_id, MY_PEER_URL, payload))
    return JSONResponse(content=make_response(
        task_id=task_id, status="queued",
        result={"output": f"任务已排队，等待处理中", "data": {"task_id": task_id}, "attachments": []},
        to_agent=MY_NAME,
    ))


async def _send_and_poll(task_id: str, peer_url: str, payload: Dict):
    """异步发任务到对端，轮询结果后更新幂等缓存"""
    await asyncio.sleep(0.5)  # 给对端一点启动时间

    cached = _idempotency.get(task_id)
    if cached:
        return

    client = get_httpx_client()
    try:
        resp = await client.post(
            f"{peer_url}/a2a",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=_TASK_TIMEOUTS.get(payload.get("payload", {}).get("task_type", "chat"), 120.0),
        )
        resp.raise_for_status()
        result = resp.json()

        if result.get("status") == "success":
            _idempotency.set(task_id, "success", result.get("result"))

        logger.info(f"[{MY_NAME}] 任务 {task_id} 已完成 status={result.get('status')}")
    except Exception as exc:
        logger.exception(f"[{MY_NAME}] 任务 {task_id} 失败: {exc}")
        _idempotency.set(task_id, "failed", {"output": f"任务执行失败: {exc}"})


# ─── Lifespan + 启动 ────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    asyncio.create_task(_cleanup_loop())


async def _cleanup_loop():
    """后台每小时清理一次过期会话"""
    while True:
        await asyncio.sleep(3600)
        try:
            _session_store.cleanup_expired()
            logger.info(f"[{MY_NAME}] 会话清理完成")
        except Exception as exc:
            logger.warning(f"[{MY_NAME}] 会话清理异常: {exc}")

# ─── 启动 ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=MY_PORT, log_level="info")
