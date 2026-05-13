#!/usr/bin/env python3
"""
Agent A — A2A Server（端口 8643）

功能：
  POST /tasks       — 接收任务，AI 对话处理，返回 task_result
  GET  /health     — 健康检查
  GET  /tasks/{id} — 任务状态查询

核心特性：
  · 事件循环不堵 — 同步代码走 asyncio.to_thread()
  · AI 异步预热 — 启动时后台加载，不阻塞 ping
  · 幂等存储    — SQLite task_id 查重
  · 状态机      — pending → processing → completed/failed

依赖：fastapi uvicorn httpx
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

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
import uvicorn

# ===== 日志 =====
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("a2a-server-a")

# ===== FastAPI App =====
app = FastAPI(title="Agent A — A2A Server", version="1.0.0")

# ===== 内存任务存储 =====
tasks: Dict[str, Dict[str, Any]] = {}

# ════════════════════════════════════════════════════════════
# 幂等存储（SQLite）
# ════════════════════════════════════════════════════════════

class IdempotencyStore:
    """task_id 查重，命中直接返回缓存结果"""

    def __init__(self, db_path: str = "/tmp/a2a_idempotency.db"):
        self._lock = threading.Lock()
        self._db = sqlite3.connect(db_path, timeout=5, check_same_thread=False)
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS processed_tasks (
                task_id TEXT PRIMARY KEY,
                status  TEXT NOT NULL,
                result  TEXT,
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
# AI Agent 懒加载 + 异步预热
# ════════════════════════════════════════════════════════════

_agent = None


@app.on_event("startup")
async def startup():
    """启动时异步预热 AI Agent，不阻塞请求处理"""
    asyncio.create_task(preload_agent())


async def preload_agent():
    """后台预加载 AI Agent"""
    global _agent
    try:
        logger.info("[A2A] AI Agent 预热中...")
        _agent = await asyncio.to_thread(_load_agent_sync)
        logger.info("[A2A] AI Agent 预热完成")
    except Exception as exc:
        logger.exception(f"[A2A] AI Agent 预热失败: {exc}")


def _load_agent_sync():
    """
    同步加载 AI Agent。
    ─────────────────────────────────────────────────────────────
    在这里替换为你实际使用的 AI 初始化逻辑。
    示例（OpenAI）：
        from openai import OpenAI
        return OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    示例（Anthropic）：
        import anthropic
        return anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    ─────────────────────────────────────────────────────────────
    """
    # ── TODO: 替换为你的 AI 初始化代码 ──
    # 示例（占位）：
    # from some_ai_sdk import AIAgent
    # return AIAgent(api_key=os.environ.get("YOUR_API_KEY"))
    return None


def get_agent():
    """返回已预热的 Agent（可能为 None）"""
    return _agent


# ════════════════════════════════════════════════════════════
# AI 调用（异步 httpx，事件循环不堵）
# ════════════════════════════════════════════════════════════

async def call_ai(system_prompt: str, instruction: str) -> str:
    """
    httpx 异步调用 LLM API。
    ─────────────────────────────────────────────────────────────
    在这里替换为你实际使用的 LLM 提供商。

    示例（OpenAI Compatible）：
        base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        api_key  = os.environ.get("OPENAI_API_KEY", "")

        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.post(
                f"{base_url}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": "gpt-4o",
                    "max_tokens": 1024,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": instruction},
                    ],
                },
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
    ─────────────────────────────────────────────────────────────
    """
    # ── TODO: 替换为你的 LLM API 调用 ──
    await asyncio.sleep(0.1)  # 避免未实现时阻塞
    return f"[AI 未实现] 收到: {instruction[:50]}"


# ════════════════════════════════════════════════════════════
# 工具函数
# ════════════════════════════════════════════════════════════

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


# ════════════════════════════════════════════════════════════
# 路由
# ════════════════════════════════════════════════════════════

@app.get("/health")
async def health():
    """健康检查"""
    return {"status": "ok", "agent": "agent-a", "port": 8643}

@app.get("/ready")
async def ready():
    """
    就绪探针：AI Agent 预热完成后返回 ok。
    用于 Kubernetes readinessProbe，避免预热期间接收流量。
    """
    agent_ready = get_agent() is not None
    return {"status": "ok" if agent_ready else "loading", "agent_ready": agent_ready}


@app.get("/live")
async def live():
    """
    存活探针：始终返回 ok。
    用于 Kubernetes livenessProbe，只要进程在就返回 ok。
    """
    return {"status": "ok"}



@app.post("/tasks")
async def create_task(request: Dict[str, Any]):
    """
    接收任务委托，AI 对话处理后返回 task_result。

    Body:
        {
          "version": "1.0",
          "type": "task_delegate",
          "from": "agent-b",
          "to": "agent-a",
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

        # task_type 可能藏在 context 里（跨 agent 消息格式）
        task_type = context.get("task_type") or payload.get("task_type", "chat")
        instruction = payload.get("instruction", "")

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

        if to_agent and to_agent != "agent-a":
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
                from_agent="agent-a",
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
                from_agent="agent-a",
                to_agent=from_agent,
            ))

        # ── 构建 system prompt ──────────────────────────────
        system_prompt = "You are Agent A, a helpful AI assistant. Be concise and direct."
        if context:
            system_prompt += f"\n\n附加上下文: {json.dumps(context, ensure_ascii=False)}"

        # ── 进 processing ─────────────────────────────────
        tasks[task_id].update({
            "status": "processing",
            "started_at": datetime.now().isoformat(),
        })

        # ── AI 调用（异步，不堵事件循环）───────────────────
        try:
            response_text = await call_ai(system_prompt, instruction)
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
            from_agent="agent-a",
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


# ════════════════════════════════════════════════════════════
# 启动
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8643, log_level="info")
