#!/usr/bin/env python3
"""
Agent A2A Server — v2.2.0
Receives task delegations and processes with AI
httpx AsyncClient 直调 AI，替代 subprocess
"""
import asyncio
import json
import logging
import os
import sqlite3
import sys
import threading
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

# 设置 PYTHONPATH 确保能导入 hermes 模块
sys.path.insert(0, os.environ.get("HERMES_PATH", "/opt/hermes"))
sys.path.insert(0, os.path.join(os.environ.get("HERMES_PATH", "/opt/hermes"), ".venv/lib/python3.11/site-packages"))

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
import httpx
import uvicorn

# ===== 日志 =====
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("a2a_server")

# ===== FastAPI App =====
app = FastAPI(title="Agent A2A Server", version="2.2.0")

# ===== 内存任务存储 =====
tasks: Dict[str, Dict[str, Any]] = {}

# ===== 幂等存储（SQLite，TTL 24h，failed 可重试）=====
class IdempotencyStore:
    """轻量幂等存储：task_id 查重 + TTL 缓存 + failed 可重试"""

    TTL_HOURS = 24

    def __init__(self, db_path: str = "/tmp/a2a_idempotency.db"):
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
        """查询 task_id 是否有未过期的缓存（completed 缓存才返回，failed 允许重试）"""
        with self._lock:
            cursor = self._db.execute(
                "SELECT status, result, created_at, completed_at FROM processed_tasks WHERE task_id = ?",
                (task_id,)
            )
            row = cursor.fetchone()
            if not row:
                return None
            status, result_json, created_at, completed_at = row
            if status == "failed":
                self._db.execute("DELETE FROM processed_tasks WHERE task_id = ?", (task_id,))
                self._db.commit()
                return None
            import datetime as dt
            created = dt.datetime.fromisoformat(created_at)
            if (dt.datetime.now() - created).total_seconds() > self.TTL_HOURS * 3600:
                self._db.execute("DELETE FROM processed_tasks WHERE task_id = ?", (task_id,))
                self._db.commit()
                return None
            return {"status": status, "result": json.loads(result_json) if result_json else None, "completed_at": completed_at}

    def set(self, task_id: str, status: str, result: Optional[Dict] = None):
        """写入处理结果"""
        now = datetime.now().isoformat()
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO processed_tasks (task_id, status, result, created_at, completed_at) VALUES (?, ?, ?, ?, ?)",
                (task_id, status, json.dumps(result, ensure_ascii=False), now, now)
            )
            self._db.commit()

    def cleanup_expired(self):
        """清理过期缓存"""
        import datetime as dt
        cutoff = (dt.datetime.now() - dt.timedelta(hours=self.TTL_HOURS)).isoformat()
        with self._lock:
            self._db.execute("DELETE FROM processed_tasks WHERE status='completed' AND created_at < ?", (cutoff,))
            self._db.commit()

_idempotency = IdempotencyStore()


# ===== 会话亲和存储（SQLite，TTL + 阈值压缩）=====
class SessionStore:
    """会话历史存储：session_id → 历史消息列表，支持多轮对话上下文

    压缩策略：当历史超过 12 条时，保留最近 3 条 + 生成摘要压缩
    """

    COMPRESS_THRESHOLD = 12
    KEEP_RECENT = 3
    TTL_DAYS = 7

    def __init__(self, db_path: str = "/tmp/a2a_sessions.db"):
        self._lock = threading.Lock()
        self._db = sqlite3.connect(db_path, timeout=5, check_same_thread=False)
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS a2a_sessions (
                session_id TEXT PRIMARY KEY,
                messages TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                is_compressed INTEGER NOT NULL DEFAULT 0
            )
        """)
        self._db.execute("CREATE INDEX IF NOT EXISTS idx_updated_at ON a2a_sessions(updated_at)")

    def get_history(self, session_id: str, limit: int = 10) -> list:
        with self._lock:
            cursor = self._db.execute(
                "SELECT messages FROM a2a_sessions WHERE session_id = ?",
                (session_id,)
            )
            row = cursor.fetchone()
            if not row:
                return []
            msgs = json.loads(row[0])
            return msgs[-limit:] if len(msgs) > limit else msgs

    def append(self, session_id: str, role: str, content: str, limit: int = 20):
        with self._lock:
            cursor = self._db.execute(
                "SELECT messages, is_compressed FROM a2a_sessions WHERE session_id = ?",
                (session_id,)
            )
            row = cursor.fetchone()
            msgs = json.loads(row[0]) if row else []
            msgs.append({"role": role, "content": content})

            if len(msgs) > self.COMPRESS_THRESHOLD:
                msgs = self._compress_rules(msgs)

            msgs = msgs[-limit:]
            self._db.execute(
                "INSERT OR REPLACE INTO a2a_sessions (session_id, messages, updated_at, is_compressed) VALUES (?, ?, ?, ?)",
                (session_id, json.dumps(msgs, ensure_ascii=False), datetime.now().isoformat(), 1)
            )
            self._db.commit()

    def _compress_rules(self, msgs: list) -> list:
        summary_parts = []
        for m in msgs[:-self.KEEP_RECENT]:
            if m.get("role") in ("user", "assistant"):
                content = m["content"]
                if len(content) > 50:
                    content = content[:50] + "..."
                summary_parts.append(f"{m['role']}: {content}")
        summary = "; ".join(summary_parts[:8])
        summary = f"[上文摘要({len(msgs)}条): {summary}]"
        system_msgs = [m for m in msgs if m.get("role") == "system"]
        recent = msgs[-self.KEEP_RECENT:]
        return system_msgs + [{"role": "system", "content": summary}] + recent

    def cleanup_expired(self):
        import datetime as dt
        cutoff = (dt.datetime.now() - dt.timedelta(days=self.TTL_DAYS)).isoformat()
        with self._lock:
            self._db.execute("DELETE FROM a2a_sessions WHERE updated_at < ?", (cutoff,))
            self._db.commit()

_session_store = SessionStore()


# ===== httpx 全局客户端（连接池复用）=======
_httpx_client: httpx.AsyncClient = None

def get_httpx_client() -> httpx.AsyncClient:
    """单例 httpx 客户端，连接池复用"""
    global _httpx_client
    if _httpx_client is None or _httpx_client.is_closed:
        _httpx_client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=5.0, read=30.0, write=10.0, pool=5.0),
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
        )
    return _httpx_client


# ===== AI 调用（异步 httpx，事件循环不堵）=====
_TASK_TIMEOUTS = {
    "ping": 5.0,
    "chat": 15.0,
    "web_search": 60.0,
    "coding": 60.0,
    "analysis": 60.0,
}

async def call_ai(system_prompt: str, instruction: str, session_id: str = None, task_type: str = "chat") -> str:
    """httpx 异步调用 AI API，session_id 相同则注入历史上下文
    按 task_type 差异化超时，错误分类重试
    """
    api_key = os.environ.get("AI_PROVIDER_API_KEY", "")
    base_url = os.environ.get("AI_PROVIDER_BASE_URL", "https://api.minimaxi.com/anthropic")
    model = os.environ.get("A2A_MODEL", "gpt-4o")
    max_tokens = int(os.environ.get("A2A_MAX_TOKENS", "1024"))

    env_path = os.environ.get("A2A_ENV_PATH", ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                if line.strip().startswith("#"):
                    continue
                if "AI_PROVIDER_API_KEY" in line:
                    api_key = line.split("=", 1)[1].strip()
                elif "AI_PROVIDER_BASE_URL" in line:
                    base_url = line.split("=", 1)[1].strip()

    timeout = _TASK_TIMEOUTS.get(task_type, 15.0)
    messages = _build_messages(instruction, session_id)
    payload = {
        "model": model,
        "max_tokens": max_tokens,
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
            output = ""
            for item in data.get("content", []):
                if item.get("type") == "text":
                    output = item.get("text", "")
            if session_id and output:
                _session_store.append(session_id, "user", instruction)
                _session_store.append(session_id, "assistant", output)
            return output

        except httpx.HTTPStatusError as exc:
            last_exc = exc
            if exc.response.status_code == 429:
                wait = 2 ** attempt
                logger.warning(f"[A2A] 429 限流，等待 {wait}s 后重试（第{attempt+1}次）")
                await asyncio.sleep(wait)
                continue
            if exc.response.status_code >= 500:
                logger.warning(f"[A2A] 服务端错误 {exc.response.status_code}，等待 1s 重试")
                await asyncio.sleep(1)
                continue
            raise

        except (httpx.ReadTimeout, httpx.ConnectError) as exc:
            last_exc = exc
            if attempt < 2:
                logger.warning(f"[A2A] 网络异常 {type(exc).__name__}，等待 1s 重试")
                await asyncio.sleep(1)
                continue
            raise

    raise last_exc or RuntimeError("AI 调用失败")


def _build_messages(instruction: str, session_id: str = None) -> list:
    """构建 AI 消息列表：如果有历史则注入"""
    messages = []
    if session_id:
        history = _session_store.get_history(session_id, limit=10)
        messages.extend(history)
    messages.append({"role": "user", "content": instruction})
    return messages


# ===== 工具函数 =====
def make_response(
    task_id: str,
    status: str,
    result: Optional[Dict] = None,
    error: Optional[Dict] = None,
    from_agent: str = "agent-a",
    to_agent: str = "agent-b",
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


# ===== 路由 =====

@app.get("/health")
async def health():
    """健康检查"""
    return {"status": "ok", "agent": "agent-a", "port": 8643}


@app.post("/tasks")
async def create_task(request: Dict[str, Any]):
    """
    接收任务委托，用 AI 对话处理
    Body: {version, type, from, to, task_id, timestamp, payload}
    """
    try:
        version = request.get("version", "1.0")
        msg_type = request.get("type")
        from_agent = request.get("from")
        to_agent = request.get("to")
        task_id = request.get("task_id") or str(uuid.uuid4())
        timestamp = request.get("timestamp", datetime.now().isoformat())
        payload = request.get("payload", {})
        context = payload.get("context", {})
        session_id = payload.get("session_id") or context.get("session_id")

        logger.info(f"[A2A] 收到任务: {task_id} from {from_agent} → {to_agent} ({msg_type}) session={session_id}")

        if msg_type != "task_delegate":
            return JSONResponse(
                status_code=400,
                content=make_response(
                    task_id=task_id,
                    status="failed",
                    error={"code": 400, "message": f"不支持的消息类型: {msg_type}"},
                ),
            )

        if to_agent and to_agent != "agent-a":
            return JSONResponse(
                status_code=400,
                content=make_response(
                    task_id=task_id,
                    status="failed",
                    error={"code": 400, "message": f"未知目标: {to_agent}"},
                ),
            )

        instruction = payload.get("instruction", "")
        context = payload.get("context", {})
        # task_type 可能藏在 context 里（跨 agent 消息格式）
        task_type = context.get("task_type") or payload.get("task_type", "chat")

        # ===== 幂等检查 =====
        cached = _idempotency.get(task_id)
        if cached:
            logger.info(f"[A2A] 幂等命中: {task_id}，直接返回缓存结果")
            return JSONResponse(content=make_response(
                task_id=task_id,
                status=cached["status"],
                result=cached["result"],
                from_agent=os.environ.get("AGENT_NAME", "agent-a"),
                to_agent=from_agent,
            ))

        # ===== 写入任务：pending =====
        tasks[task_id] = {
            "status": "pending",
            "created_at": datetime.now().isoformat(),
            "from": from_agent,
            "task_type": task_type,
            "instruction": instruction,
            "response": None,
            "started_at": None,
            "completed_at": None,
        }

        # ===== ping 直接返回，不走 AI =====
        if task_type == "ping":
            tasks[task_id].update({
                "status": "completed",
                "response": "pong",
                "completed_at": datetime.now().isoformat(),
            })
            result = {"output": "pong", "data": {"task_id": task_id, "session_id": session_id}, "attachments": []}
            _idempotency.set(task_id, "completed", result)
            return JSONResponse(content=make_response(
                task_id=task_id,
                status="completed",
                result=result,
                from_agent=os.environ.get("AGENT_NAME", "agent-a"),
                to_agent=from_agent,
            ))

        # ===== 构建 system prompt =====
        system_prompt = "你是 Agent A，一个 AI 助手。回答简洁直接，一针见血。"
        if context:
            system_prompt += f"\n\n附加上下文: {json.dumps(context, ensure_ascii=False)}"

        # ===== 进 processing =====
        tasks[task_id].update({
            "status": "processing",
            "started_at": datetime.now().isoformat(),
        })

        # ===== AI 调用（异步，不堵事件循环）=====
        try:
            response_text = await call_ai(system_prompt, instruction, session_id, task_type)
        except Exception as ai_exc:
            import sys
            print(f"[A2A DEBUG] AI exception: {type(ai_exc).__name__}: {ai_exc}", file=sys.stderr)
            response_text = f"收到: {instruction[:50]}... (AI处理异常)"

        logger.info(f"[A2A] AI 响应: {response_text[:80]}...")

        result = {
            "output": response_text,
            "data": {"task_id": task_id, "session_id": session_id},
            "attachments": [],
        }

        # ===== 完成任务 =====
        tasks[task_id].update({
            "status": "completed",
            "response": response_text,
            "completed_at": datetime.now().isoformat(),
        })
        _idempotency.set(task_id, "completed", result)

        return JSONResponse(content=make_response(
            task_id=task_id,
            status="success",
            result=result,
            from_agent=os.environ.get("AGENT_NAME", "agent-a"),
            to_agent=from_agent,
        ))

    except Exception as exc:
        logger.exception(f"[A2A] 处理任务异常: {exc}")
        return JSONResponse(
            status_code=500,
            content=make_response(
                task_id=request.get("task_id", "unknown"),
                status="failed",
                error={"code": 500, "message": str(exc)},
            ),
        )


@app.get("/tasks/{task_id}")
async def get_task(task_id: str):
    """查询任务状态"""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail=f"任务 {task_id} 不存在")
    return tasks[task_id]


# ===== 启动 =====
if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.environ.get("A2A_PORT", "8643")),
        log_level="info",
    )
