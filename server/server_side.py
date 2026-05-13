#!/usr/bin/env python3
"""
Agent A2A Server Template
Receives task delegations and processes with AI
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
import uvicorn

# ===== 日志 =====
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("a2a_server")

# ===== FastAPI App =====
app = FastAPI(title="Agent A2A Server", version="2.1.0")

# ===== 内存任务存储 =====
tasks: Dict[str, Dict[str, Any]] = {}

# ===== 幂等存储（SQLite）=====
class IdempotencyStore:
    """轻量幂等存储：task_id 查重 + 结果缓存"""

    def __init__(self, db_path: str = "/tmp/a2a_idempotency.db"):
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
        """查询 task_id 是否已有结果"""
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
        """写入处理结果"""
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO processed_tasks (task_id, status, result, completed_at) VALUES (?, ?, ?, ?)",
                (task_id, status, json.dumps(result, ensure_ascii=False), datetime.now().isoformat())
            )
            self._db.commit()

_idempotency = IdempotencyStore()


# ===== AI Agent 初始化（启动预热，不阻塞请求）=====
_agent = None

@app.on_event("startup")
async def startup():
    """启动时异步预热 AI Agent，不阻塞请求"""
    asyncio.create_task(preload_agent())

async def preload_agent():
    """后台预加载 AI Agent"""
    global _agent
    try:
        logger.info("[A2A] 预热 AI Agent 中...")
        _agent = await asyncio.to_thread(load_agent_sync)
        logger.info("[A2A] AI Agent 预热完成")
    except Exception as exc:
        logger.exception(f"[A2A] AI Agent 预热失败: {exc}")

def load_agent_sync():
    """同步加载 AIAgent（运行在线程池中）"""
    from run_agent import AIAgent
    from hermes_cli.config import load_config

    config = load_config()
    model_cfg = config.get("model", {})

    api_key = os.environ.get("OPENAI_API_KEY", "")
    base_url = os.environ.get("OPENAI_BASE_URL", "") or model_cfg.get("base_url", "")
    model = model_cfg.get("model", "gpt-4o")
    provider = model_cfg.get("provider", "openai")

    return AIAgent(
        base_url=base_url,
        api_key=api_key,
        provider=provider,
        model=model,
        platform="a2a",
        skip_memory=False,
        skip_context_files=True,
    )

def get_agent():
    """懒加载，返回已预热或已加载的 Agent"""
    return _agent


# ===== AI 调用（异步 httpx，事件循环不堵）=====
async def call_ai(system_prompt: str, instruction: str) -> str:
    """httpx 异步调用 MiniMax API，事件循环始终保持响应"""
    import httpx

    api_key = ""
    base_url = os.environ.get("AI_PROVIDER_BASE_URL", "https://api.minimaxi.com/anthropic")
    env_path = os.environ.get("A2A_ENV_PATH", ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                if "AI_PROVIDER_API_KEY" in line and not line.strip().startswith("#"):
                    api_key = line.split("=", 1)[1].strip()
                    break
                if "AI_PROVIDER_BASE_URL" in line and not line.strip().startswith("#"):
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
                "model": os.environ.get("A2A_MODEL", "gpt-4o"),
                "max_tokens": int(os.environ.get("A2A_MAX_TOKENS", "1024")),
                "system": system_prompt,
                "messages": [{"role": "user", "content": instruction}],
            },
        )
        resp.raise_for_status()
        data = resp.json()
        for item in data.get("content", []):
            if item.get("type") == "text":
                return item.get("text", "")
        return str(data.get("content", []))


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

        logger.info(f"[A2A] 收到任务: {task_id} from {from_agent} → {to_agent} ({msg_type})")

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
            result = {"output": "pong", "data": {"task_id": task_id}, "attachments": []}
            _idempotency.set(task_id, "completed", result)
            return JSONResponse(content=make_response(
                task_id=task_id,
                status="completed",
                result=result,
                from_agent=os.environ.get("AGENT_NAME", "agent-a"),
                to_agent=from_agent,
            ))

        # ===== 构建 system prompt =====
        system_prompt = "你叫小马，是 Sid 的 AI 助手。回答简洁直接，一针见血。"
        if context:
            system_prompt += f"\n\n附加上下文: {json.dumps(context, ensure_ascii=False)}"

        # ===== 进 processing =====
        tasks[task_id].update({
            "status": "processing",
            "started_at": datetime.now().isoformat(),
        })

        # ===== AI 调用（异步，不堵事件循环）=====
        try:
            response_text = await call_ai(system_prompt, instruction)
        except Exception as ai_exc:
            import sys
            print(f"[A2A DEBUG] AI exception: {type(ai_exc).__name__}: {ai_exc}", file=sys.stderr)
            response_text = f"收到: {instruction[:50]}... (AI处理异常)"

        logger.info(f"[A2A] AI 响应: {response_text[:80]}...")

        result = {
            "output": response_text,
            "data": {"task_id": task_id},
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
