#!/usr/bin/env python3
"""
Agent B A2A Server — v2.0.0
Receives task delegations from Agent A, processes with AI, returns results.
脱敏版：环境变量控制所有敏感配置
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

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
import uvicorn

# ===== 日志 =====
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("agent_b_a2a")

# ===== FastAPI App =====
app = FastAPI(
    title="Agent B A2A Server",
    version="2.0.0",
    description="Receives task delegations and processes with AI"
)

# ===== 内存任务存储 =====
tasks: Dict[str, Dict[str, Any]] = {}

# ===== 幂等存储（SQLite）=====
class IdempotencyStore:
    """轻量幂等存储：task_id 查重 + 结果缓存"""

    def __init__(self, db_path: str = "/tmp/agent_b_a2a_idempotency.db"):
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
        self._db.execute("CREATE INDEX IF NOT EXISTS idx_completed_at ON processed_tasks(completed_at)")

    def get(self, task_id: str) -> Optional[Dict]:
        with self._lock:
            cursor = self._db.execute(
                "SELECT status, result, completed_at FROM processed_tasks WHERE task_id = ?",
                (task_id,)
            )
            row = cursor.fetchone()
            if row:
                return {"status": row[0], "result": json.loads(row[1]) if row[1] else None, "completed_at": row[2]}
            return None

    def set(self, task_id: str, status: str, result: Optional[Dict] = None):
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO processed_tasks (task_id, status, result, completed_at) VALUES (?, ?, ?, ?)",
                (task_id, status, json.dumps(result, ensure_ascii=False), datetime.now().isoformat())
            )
            self._db.commit()

_idempotency = IdempotencyStore(
    db_path=os.environ.get("A2A_IDEMPOTENCY_DB", "/tmp/agent_b_a2a_idempotency.db")
)


# ===== 会话亲和存储（SQLite）=====
class SessionStore:
    """会话历史存储：session_id → 历史消息列表，支持多轮对话上下文"""

    def __init__(self, db_path: str = "/tmp/agent_b_a2a_sessions.db"):
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
                "SELECT messages FROM a2a_sessions WHERE session_id = ?",
                (session_id,)
            )
            row = cursor.fetchone()
            msgs = json.loads(row[0]) if row else []
            msgs.append({"role": role, "content": content})
            msgs = msgs[-limit:]
            self._db.execute(
                "INSERT OR REPLACE INTO a2a_sessions (session_id, messages, updated_at) VALUES (?, ?, ?)",
                (session_id, json.dumps(msgs, ensure_ascii=False), datetime.now().isoformat())
            )
            self._db.commit()

_session_store = SessionStore()


# ===== AI 调用（异步 httpx，事件循环不堵）=====
async def call_ai(system_prompt: str, instruction: str, session_id: str = None) -> str:
    """httpx 异步调用 AI API，session_id 相同则注入历史上下文"""
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

    async with httpx.AsyncClient(timeout=20.0) as client:
        resp = await client.post(
            f"{base_url}/v1/messages",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
            },
            json={
                "model": model,
                "max_tokens": max_tokens,
                "system": system_prompt,
                "messages": _build_messages(instruction, session_id),
            },
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
    from_agent: str = "agent-b",
    to_agent: str = "agent-a",
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
    return {
        "status": "ok",
        "agent": os.environ.get("AGENT_NAME", "agent-b"),
        "port": int(os.environ.get("A2A_PORT", "8644"))
    }

@app.post("/tasks")
async def create_task(request: Dict[str, Any]):
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

        if to_agent and to_agent != os.environ.get("AGENT_NAME", "agent-b"):
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
        task_type = context.get("task_type") or payload.get("task_type", "chat")

        # ===== 幂等检查 =====
        cached = _idempotency.get(task_id)
        if cached:
            logger.info(f"[A2A] 幂等命中: {task_id}")
            return JSONResponse(content=make_response(
                task_id=task_id,
                status=cached["status"],
                result=cached["result"],
                from_agent=os.environ.get("AGENT_NAME", "agent-b"),
                to_agent=from_agent,
            ))

        # ===== 写入任务 =====
        tasks[task_id] = {
            "status": "pending",
            "created_at": datetime.now().isoformat(),
            "from": from_agent,
            "task_type": task_type,
            "instruction": instruction,
        }

        # ===== ping 直接返回 =====
        if task_type == "ping":
            tasks[task_id]["status"] = "completed"
            result = {"output": "pong", "data": {"task_id": task_id, "session_id": session_id}, "attachments": []}
            _idempotency.set(task_id, "completed", result)
            return JSONResponse(content=make_response(
                task_id=task_id,
                status="success",
                result=result,
                from_agent=os.environ.get("AGENT_NAME", "agent-b"),
                to_agent=from_agent,
            ))

        # ===== 构建 system prompt =====
        system_prompt = f"你是 Agent B，一个 AI 助手。回答简洁直接，一针见血。"
        if context:
            system_prompt += f"\n\n附加上下文: {json.dumps(context, ensure_ascii=False)}"

        # ===== AI 调用 =====
        tasks[task_id]["status"] = "processing"
        try:
            response_text = await call_ai(system_prompt, instruction, session_id)
        except Exception as ai_exc:
            logger.exception(f"[A2A] AI 调用异常: {ai_exc}")
            response_text = f"收到: {instruction[:50]}... (AI处理异常)"

        logger.info(f"[A2A] AI 响应: {response_text[:80]}...")

        result = {"output": response_text, "data": {"task_id": task_id, "session_id": session_id}, "attachments": []}
        tasks[task_id]["status"] = "completed"
        _idempotency.set(task_id, "success", result)

        return JSONResponse(content=make_response(
            task_id=task_id,
            status="success",
            result=result,
            from_agent=os.environ.get("AGENT_NAME", "agent-b"),
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
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail=f"任务 {task_id} 不存在")
    return tasks[task_id]

# ===== 启动 =====
if __name__ == "__main__":
    port = int(os.environ.get("A2A_PORT", "8644"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
