#!/usr/bin/env python3
"""
Agent B — A2A Server（端口 8644）

功能：
  POST /a2a       — 接收 Agent A 的任务，AI 对话处理，返回 task_result
  GET  /health    — 健康检查
  GET  /tasks/{id} — 任务状态查询

核心特性：
  · 事件循环不堵 — 同步代码走 asyncio.to_thread()
  · 幂等存储    — SQLite task_id 查重
  · 状态机      — pending → processing → completed/failed

依赖：fastapi uvicorn
"""

import asyncio
import json
import logging
import os
import sqlite3
import subprocess
import sys
import threading
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import uvicorn

# ===== 日志 =====
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("a2a-server-b")

# ===== FastAPI App =====
app = FastAPI(title="Agent B — A2A Server", version="1.0.0")

# ===== 内存任务存储 =====
tasks: Dict[str, Dict[str, Any]] = {}

# ════════════════════════════════════════════════════════════
# 幂等存储（SQLite）
# ════════════════════════════════════════════════════════════

class IdempotencyStore:
    def __init__(self, db_path: str = "/tmp/a2a_b_idempotency.db"):
        self._lock = threading.Lock()
        self._db = sqlite3.connect(db_path, timeout=5, check_same_thread=False)
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS processed_tasks (
                task_id     TEXT PRIMARY KEY,
                status      TEXT NOT NULL,
                result      TEXT,
                completed_at TEXT NOT NULL
            )
        """)
        self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_completed_at ON processed_tasks(completed_at)"
        )

    def get(self, task_id: str) -> Optional[Dict]:
        with self._lock:
            row = self._db.execute(
                "SELECT status, result, completed_at FROM processed_tasks WHERE task_id = ?",
                (task_id,),
            ).fetchone()
            if row:
                return {
                    "status": row[0],
                    "result": json.loads(row[1]) if row[1] else None,
                    "completed_at": row[2],
                }
            return None

    def set(self, task_id: str, status: str, result: Optional[Dict] = None):
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO processed_tasks VALUES (?, ?, ?, ?)",
                (task_id, status, json.dumps(result, ensure_ascii=False),
                 datetime.now().isoformat()),
            )
            self._db.commit()


_idempotency = IdempotencyStore()

# ════════════════════════════════════════════════════════════
# AI 调用（本例使用 subprocess 调用本地 AI CLI）
# ════════════════════════════════════════════════════════════

def run_ai_chat_sync(message: str, timeout: int = 120) -> str:
    """
    同步调用 AI 的方式。
    ─────────────────────────────────────────────────────────────
    在这里替换为你实际使用的 AI 调用方式。

    方式 1 — 本地 CLI（示例）：
        result = subprocess.run(
            ["your-ai-cli", "agents", "chat", "--message", message],
            capture_output=True, text=True, timeout=timeout
        )
        return parse_cli_output(result.stdout)

    方式 2 — HTTP API（示例）：
        import httpx, asyncio
        async def call():
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(
                    "https://api.your-provider.com/v1/chat",
                    headers={"Authorization": f"Bearer {os.environ['API_KEY']}"},
                    json={"model": "xxx", "messages": [{"role":"user","content":message}]}
                )
                return resp.json()["choices"][0]["message"]["content"]
        return asyncio.run(call())
    ─────────────────────────────────────────────────────────────
    """
    # ── TODO: 替换为你的 AI 调用 ──
    return f"[AI 未实现] 收到: {message[:50]}"


async def call_ai_async(message: str, timeout: int = 120) -> str:
    """
    AI 调用入口 — 同步代码走线程池，事件循环始终保持响应。
    """
    return await asyncio.to_thread(run_ai_chat_sync, message, timeout)


# ════════════════════════════════════════════════════════════
# 工具函数
# ════════════════════════════════════════════════════════════

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


# ════════════════════════════════════════════════════════════
# 路由
# ════════════════════════════════════════════════════════════

@app.get("/health")
async def health():
    return {"status": "ok", "agent": "agent-b", "port": 8644}


@app.post("/a2a")
async def create_task(request: Dict[str, Any]):
    """
    接收 Agent A 的任务委托，AI 对话处理后返回 task_result。

    Body:
        {
          "version": "1.0",
          "type": "task_delegate",
          "from": "agent-a",
          "to": "agent-b",
          "task_id": "uuid-v4",
          "timestamp": "...",
          "payload": {
            "task_type": "chat | ping | ...",
            "instruction": "...",
            "context": {}
          }
        }
    """
    try:
        version    = request.get("version", "1.0")
        msg_type   = request.get("type")
        from_agent = request.get("from", "unknown")
        to_agent   = request.get("to", "unknown")
        task_id    = request.get("task_id") or str(uuid.uuid4())
        timestamp  = request.get("timestamp", datetime.now().isoformat())
        payload    = request.get("payload", {})
        context    = payload.get("context", {})

        # task_type 可能藏在 context 里
        task_type    = context.get("task_type") or payload.get("task_type", "chat")
        instruction  = payload.get("instruction", "")

        logger.info(
            f"[A2A] 收到任务: {task_id} | {from_agent} → {to_agent} | type={task_type}"
        )

        # ── 消息类型过滤 ─────────────────────────────────────
        if msg_type != "task_delegate":
            return JSONResponse(
                status_code=400,
                content=make_response(
                    task_id=task_id,
                    status="failed",
                    error={"code": 400, "message": f"不支持的消息类型: {msg_type}"},
                ),
            )

        if to_agent and to_agent != "agent-b":
            return JSONResponse(
                status_code=400,
                content=make_response(
                    task_id=task_id,
                    status="failed",
                    error={"code": 400, "message": f"未知目标: {to_agent}"},
                ),
            )

        # ── 幂等检查 ────────────────────────────────────────
        cached = _idempotency.get(task_id)
        if cached:
            logger.info(f"[A2A] 幂等命中: {task_id}")
            return JSONResponse(content=make_response(
                task_id=task_id,
                status=cached["status"],
                result=cached["result"],
                from_agent="agent-b",
                to_agent=from_agent,
            ))

        # ── 写入任务：pending ────────────────────────────────
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

        # ── ping：直接返回，不走 AI ─────────────────────────
        if task_type == "ping":
            tasks[task_id].update({
                "status": "completed",
                "response": "pong",
                "completed_at": datetime.now().isoformat(),
            })
            result = {"output": "pong", "data": {"task_id": task_id}, "attachments": []}
            _idempotency.set(task_id, "completed", result)
            return JSONResponse(content=make_response(
                task_id=task_id,
                status="completed",
                result=result,
                from_agent="agent-b",
                to_agent=from_agent,
            ))

        # ── 进 processing ─────────────────────────────────
        tasks[task_id].update({
            "status": "processing",
            "started_at": datetime.now().isoformat(),
        })

        # ── AI 调用（异步，不堵事件循环）───────────────────
        try:
            response_text = await call_ai_async(instruction)
        except Exception as ai_exc:
            logger.warning(f"[A2A] AI 调用异常: {type(ai_exc).__name__}: {ai_exc}")
            response_text = f"收到: {instruction[:50]}... (AI处理异常)"

        logger.info(f"[A2A] AI 响应: {response_text[:80]}...")

        result = {
            "output": response_text,
            "data": {"task_id": task_id},
            "attachments": [],
        }

        # ── 完成任务 ──────────────────────────────────────
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
            from_agent="agent-b",
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


# ════════════════════════════════════════════════════════════
# 启动
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8644, log_level="info")
