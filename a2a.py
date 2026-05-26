#!/usr/bin/env python3
"""
A2A Bridge — 通用双向 Agent 通信节点 v3.0.0

设计原则：
- 每个实例既是 Server（接收任务）也是 Client（发送任务）
- 不分客户端/服务端，只有 port + peer 配置
- 环境变量驱动，所有敏感配置通过 env 注入

启动：
    A2A_PORT=8643 A2A_PEER_URL=http://localhost:8644 python a2a.py

环境变量：
    A2A_PORT              监听端口（默认 8643）
    A2A_PEER_URL          对端地址，用于主动发任务（默认空，不主动发送）
    A2A_PEER_API_KEY      对端 AI API Key（用于调用对端 AI）
    A2A_NAME              节点名称（默认从 hostname 或 "node-{port}"）
    AI_PROVIDER_API_KEY   本节点 AI API Key（用于处理收到的任务）
    AI_PROVIDER_BASE_URL  AI API 地址（默认 https://api.minimaxi.com/anthropic）
    A2A_MODEL             AI 模型（默认 gpt-4o）
    A2A_MAX_TOKENS        最大 token 数（默认 1024）
    A2A_IDEMPOTENCY_DB    幂等 DB 路径（默认 /tmp/a2a_idempotency.db）
    A2A_SESSION_DB        会话 DB 路径（默认 /tmp/a2a_sessions.db）
    A2A_ENV_PATH          .env 文件路径（用于 docker 挂载 .env）
"""
import asyncio
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

# ─── 日志 ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("a2a")

# ─── 配置读取 ─────────────────────────────────────────────────────────

def _read_env(key: str, default: str = "") -> str:
    """从环境变量或 .env 文件读取配置"""
    val = os.environ.get(key, default)
    # 尝试从 .env 覆盖（支持 docker 挂载）
    env_path = os.environ.get("A2A_ENV_PATH", ".env")
    if os.path.exists(env_path) and key not in os.environ:
        with open(env_path) as f:
            for line in f:
                ls = line.strip()
                if ls.startswith("#") or "=" not in ls:
                    continue
                k, v = ls.split("=", 1)
                if k.strip() == key:
                    val = v.strip()
                    break
    return val

# 节点自身配置
MY_PORT = int(os.environ.get("A2A_PORT", "8643"))
MY_NAME = os.environ.get("A2A_NAME", "") or os.environ.get("HOSTNAME", f"node-{MY_PORT}")
MY_PEER_URL = os.environ.get("A2A_PEER_URL", "").rstrip("/")  # 对端地址，用于主动发送

# AI 配置（用于处理收到的任务）
AI_API_KEY = os.environ.get("AI_PROVIDER_API_KEY", "")
AI_BASE_URL = os.environ.get("AI_PROVIDER_BASE_URL", "https://api.minimaxi.com/anthropic")
AI_MODEL = os.environ.get("A2A_MODEL", "gpt-4o")
AI_MAX_TOKENS = int(os.environ.get("A2A_MAX_TOKENS", "1024"))

# 数据库路径
IDEMPOTENCY_DB = os.environ.get("A2A_IDEMPOTENCY_DB", f"/tmp/a2a_idempotency_{MY_PORT}.db")
SESSION_DB = os.environ.get("A2A_SESSION_DB", f"/tmp/a2a_sessions_{MY_PORT}.db")

logger.info(f"[{MY_NAME}] 启动 A2A Bridge，port={MY_PORT} peer={MY_PEER_URL or '(仅接收)'}")

# ─── FastAPI App ──────────────────────────────────────────────────────
app = FastAPI(
    title=f"A2A Bridge — {MY_NAME}",
    version="1.0.0",
    description="通用双向 Agent 通信节点",
)

# 内存任务存储（简短状态查询）
tasks: Dict[str, Dict[str, Any]] = {}

# ─── httpx 全局客户端（连接池复用）───────────────────────────────────
_httpx_client: httpx.AsyncClient = None

def get_httpx_client() -> httpx.AsyncClient:
    global _httpx_client
    if _httpx_client is None or _httpx_client.is_closed:
        _httpx_client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=5.0, read=30.0, write=10.0, pool=5.0),
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
            if status == "failed":
                self._db.execute("DELETE FROM processed_tasks WHERE task_id = ?", (task_id,))
                self._db.commit()
                return None
            created = datetime.fromisoformat(created_at)
            if (datetime.now() - created).total_seconds() > self.TTL_HOURS * 3600:
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
        cutoff = (datetime.now() - __import__("datetime").timedelta(days=self.TTL_DAYS)).isoformat()
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

def _load_ai_key() -> str:
    """尝试从环境变量或 .env 文件加载 AI key"""
    key = os.environ.get("AI_PROVIDER_API_KEY", "")
    env_path = os.environ.get("A2A_ENV_PATH", ".env")
    if not key and os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                ls = line.strip()
                if ls.startswith("#") or "=" not in ls:
                    continue
                k, v = ls.split("=", 1)
                if k.strip() in ("AI_PROVIDER_API_KEY", "MINIMAX_CN_API_KEY"):
                    key = v.strip()
                    break
    return key


async def call_ai(
    system_prompt: str,
    instruction: str,
    session_id: str = None,
    task_type: str = "chat",
    api_key: str = "",
    base_url: str = "",
    model: str = "",
) -> str:
    """异步 httpx 调用 AI，支持多来源 key 读取"""
    if not api_key:
        api_key = _load_ai_key()
    if not base_url:
        base_url = AI_BASE_URL
    if not model:
        model = AI_MODEL

    timeout = _TASK_TIMEOUTS.get(task_type, 120.0)
    messages = _build_messages(instruction, session_id)
    payload = {
        "model": model,
        "max_tokens": AI_MAX_TOKENS,
        "system": system_prompt,
        "messages": messages,
    }

    last_exc = None
    for attempt in range(3):
        try:
            client = get_httpx_client()
            resp = await client.post(
                f"{base_url}/v1/messages",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "x-api-key": api_key,
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


def _build_messages(instruction: str, session_id: str = None) -> list:
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


# ─── HTTP 端点 ────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "name": MY_NAME,
        "port": MY_PORT,
        "peer": MY_PEER_URL or None,
        "version": "1.0.0",
    }


@app.get("/ready")
async def ready():
    """就绪探针（K8s readiness）"""
    return {"status": "ready", "name": MY_NAME}


@app.get("/live")
async def live():
    """存活探针（K8s liveness）"""
    return {"status": "ok"}


@app.get("/capabilities")
async def capabilities():
    """返回支持的特性"""
    return {
        "version": "1.0.0",
        "features": {
            "idempotency": True,
            "session_affinity": True,
            "async_preload": True,
            "capabilities_endpoint": True,
            "version_negotiation": True,
        },
    }


@app.post("/a2a")
async def receive_task(request: Dict[str, Any]):
    """接收来自对端的任务"""
    try:
        task_id = request.get("task_id") or str(uuid.uuid4())
        from_agent = request.get("from", "unknown")
        to_agent = request.get("to", "*")
        msg_type = request.get("type")
        timestamp = request.get("timestamp", datetime.now(timezone.utc).isoformat())
        payload = request.get("payload", {})
        context = payload.get("context", {})
        session_id = payload.get("session_id") or context.get("session_id")

        logger.info(f"[{MY_NAME}] ← 任务 {task_id} from {from_agent} → {to_agent} ({msg_type})")

        if msg_type != "task_delegate":
            return JSONResponse(
                status_code=400,
                content=make_response(
                    task_id=task_id, status="failed",
                    error={"code": 400, "message": f"不支持的消息类型: {msg_type}"},
                    to_agent=from_agent,
                ),
            )

        # 目标过滤（可接收 * 或 本节点名）
        if to_agent not in (MY_NAME, "*", ""):
            return JSONResponse(
                status_code=400,
                content=make_response(
                    task_id=task_id, status="failed",
                    error={"code": 400, "message": f"未知目标: {to_agent}"},
                    to_agent=from_agent,
                ),
            )

        instruction = payload.get("instruction", "")
        task_type = _resolve_task_type(payload, context)

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

        # 写入内存任务
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
            result = {"output": "pong", "data": {"task_id": task_id, "session_id": session_id}, "attachments": []}
            _idempotency.set(task_id, "completed", result)
            return JSONResponse(content=make_response(
                task_id=task_id, status="success", result=result, to_agent=from_agent,
            ))

        # ── 构建 system prompt ──
        system_prompt = f"你是 {MY_NAME}，一个 AI 助手。回答简洁直接，一针见血。"
        if context:
            system_prompt += f"\n\n附加上下文: {json.dumps(context, ensure_ascii=False)}"

        # ── AI 调用 ──
        tasks[task_id]["status"] = "processing"
        try:
            response_text = await call_ai(
                system_prompt, instruction, session_id, task_type,
                api_key=_load_ai_key(),
            )
        except Exception as exc:
            logger.exception(f"[{MY_NAME}] AI 调用异常: {exc}")
            response_text = f"(AI 处理异常: {exc})"

        logger.info(f"[{MY_NAME}] AI 响应: {response_text[:80]}...")

        result = {"output": response_text, "data": {"task_id": task_id, "session_id": session_id}, "attachments": []}
        tasks[task_id]["status"] = "completed"
        _idempotency.set(task_id, "success", result)

        return JSONResponse(content=make_response(
            task_id=task_id, status="success", result=result, to_agent=from_agent,
        ))

    except Exception as exc:
        logger.exception(f"[{MY_NAME}] 处理任务异常: {exc}")
        return JSONResponse(
            status_code=500,
            content=make_response(
                task_id=request.get("task_id", "unknown"),
                status="failed",
                error={"code": 500, "message": str(exc)},
                to_agent=request.get("from", "unknown"),
            ),
        )


@app.get("/a2a/{task_id}")
async def get_task(task_id: str):
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail=f"任务 {task_id} 不存在")
    return tasks[task_id]


@app.post("/send")
async def send_task(request: Dict[str, Any]):
    """主动向对端发送任务（用于主动发起的场景）"""
    if not MY_PEER_URL:
        return JSONResponse(
            status_code=400,
            content={"error": "A2A_PEER_URL 未配置，无法主动发送任务"},
        )
    try:
        task_id = request.get("task_id") or str(uuid.uuid4())
        task_type = request.get("task_type", "chat")
        instruction = request.get("instruction", "")
        context = request.get("context", {})
        session_id = request.get("session_id")

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
            },
        }

        # 幂等检查（主动发送也查重）
        cached = _idempotency.get(task_id)
        if cached:
            logger.info(f"[{MY_NAME}] 主动发送幂等命中 {task_id}")
            return JSONResponse(content={"status": cached["status"], "result": cached["result"], "task_id": task_id})

        timeout = _TASK_TIMEOUTS.get(task_type, 120.0)
        client = get_httpx_client()

        resp = await client.post(
            f"{MY_PEER_URL}/a2a",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=timeout,
        )
        resp.raise_for_status()
        result = resp.json()

        if result.get("status") == "success":
            _idempotency.set(task_id, "success", result.get("result"))

        return result

    except Exception as exc:
        logger.exception(f"[{MY_NAME}] 主动发送任务异常: {exc}")
        return JSONResponse(
            status_code=500,
            content={"error": str(exc), "task_id": task_id},
        )


@app.on_event("startup")
async def startup():
    """启动时异步清理过期会话"""
    asyncio.create_task(_cleanup_task())


async def _cleanup_task():
    """后台定期清理过期会话（每小时一次）"""
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