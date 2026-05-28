# A2A Bridge — 通用双向 Agent 通信框架 v1.2.2

[![Version](https://img.shields.io/badge/version-v1.2.2-blue)](https://github.com/sctale/a2a-bridge/releases)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9+-blue)](https://www.python.org/)

> 让任意两个 AI Agent 通过 HTTP+JSON 互相通信，无需依赖第三方 A2A 库。
> **一体化开箱即用**：每个实例既是 Server 也是 Client，只配一个端口和对方地址即可。

---

## 核心特性

- **一体化设计**：不区分 client/server，每个节点既能接收也能主动发送任务
- **环境变量驱动**：所有配置通过 env 注入，不需要硬编码
- **AI Key 从环境变量读取**：通过 `AI_PROVIDER_API_KEY` 或 `.env` 文件提供，无需在请求头携带
- **ping 无需 instruction**：`task_type=ping` 时 instruction 可为空
- **幂等**：SQLite task_id 查重，24h TTL，failed 自动删除可重试
- **会话亲和**：session_id 相同则自动注入历史上下文，支持多轮对话
- **健康探针**：/health + /ready（K8s）+ /live + /capabilities
- **差异化超时**：ping 5s，其他 120s，429 指数退避重试
- **同步/异步双模式**：`sync=true`（默认）同步等待结果，`sync=false` 立即返回 queued
- **reply_to 回调**：任务携带发送方回调地址，对端完成后可 POST /report 汇报结果
- **/report 端点**：接收对端汇报，回写 idempotency 缓存，支持完整委托→汇报流程

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

`requirements.txt` 内容：
```
fastapi>=0.100.0
uvicorn>=0.23.0
httpx>=0.24.0
```

### 2. 启动节点 A（端口 8643）

```bash
A2A_PORT=8643 \
A2A_PEER_URL=http://localhost:8644 \
AI_PROVIDER_API_KEY=$MINIMAX_CN_API_KEY \
python a2a.py
```

### 3. 启动节点 B（端口 8644）

```bash
A2A_PORT=8644 \
A2A_NAME=node-b \
A2A_PEER_URL=http://localhost:8643 \
AI_PROVIDER_API_KEY=$MINIMAX_CN_API_KEY \
python a2a.py
```

### 4. 验证

```bash
# 健康检查
curl http://localhost:8643/health

# Agent Card
curl http://localhost:8643/.well-known/agent-card.json

# ping 测试（本地，ping 不需要 instruction）
curl -X POST http://localhost:8643/a2a \
  -H "Content-Type: application/json" \
  -d '{
    "version": "1.0",
    "type": "task_delegate",
    "from": "test",
    "task_id": "p1",
    "payload": {"task_type": "ping", "instruction": ""}
  }'

# chat 测试（需要 AI Key）
curl -X POST http://localhost:8643/a2a \
  -H "Content-Type: application/json" \
  -d '{
    "version": "1.0",
    "type": "task_delegate",
    "from": "test",
    "task_id": "c1",
    "payload": {
      "task_type": "chat",
      "instruction": "1+1等于几？只回答数字。",
      "session_id": "sess-001"
    }
  }'

# A → B：跨节点发送（通过 /send 端点）
curl -X POST http://localhost:8643/send \
  -H "Content-Type: application/json" \
  -d '{"task_type": "ping", "instruction": ""}'
```

---

## Agent Card 机制

A2A Protocol 规定每个 Agent 必须公开自己的元数据（Agent Card），客户端通过 Agent Card 发现 Agent。

### Agent Card 端点

| 路径 | 说明 |
|------|------|
| `GET /.well-known/agent-card.json` | 官方 Well-Known 标准路径 |
| `GET /a2a/.well-known/agent-card.json` | A2A 官方路径别名（兼容官方客户端） |

### Agent Card 示例

```json
{
  "name": "hermes-nas",
  "description": "A2A Bridge Agent — hermes-nas，支持 ping/chat/web_search/coding/analysis 任务类型",
  "url": "http://localhost:8643",
  "version": "1.2.1",
  "capabilities": {
    "streaming": false,
    "pushNotifications": false,
    "extendedAgentCard": false
  },
  "skills": [
    {"id": "chat", "name": "AI 对话", "description": "通用 AI 对话助手"},
    {"id": "web_search", "name": "网络搜索", "description": "互联网信息搜索与摘要"},
    {"id": "coding", "name": "代码开发", "description": "代码编写、调试与优化"},
    {"id": "analysis", "name": "分析报告", "description": "数据与文本深度分析"}
  ]
}
```

---

## 配置说明

| 环境变量 | 默认值 | 说明 |
|---------|--------|------|
| `A2A_PORT` | `8643` | 监听端口 |
| `A2A_PEER_URL` | 空 | 对端地址（用于主动发送，不配则只收不发） |
| `A2A_NAME` | `HOSTNAME` | 节点名称，响应中标识身份 |
| `AI_PROVIDER_API_KEY` | — | AI API Key（从环境变量或 `.env` 读取） |
| `AI_PROVIDER_BASE_URL` | `https://api.minimaxi.com/anthropic` | AI API 地址 |
| `A2A_MODEL` | `gpt-4o` | AI 模型 |
| `A2A_MAX_TOKENS` | `2048` | 最大 token 数 |
| `A2A_IDEMPOTENCY_DB` | `/tmp/a2a_idempotency_{port}.db` | 幂等数据库路径 |
| `A2A_SESSION_DB` | `/tmp/a2a_sessions_{port}.db` | 会话数据库路径 |
| `A2A_ENV_PATH` | `.env` | .env 文件路径（docker 挂载用） |

---

## 协议格式

### task_delegate（任务委托）

```json
{
  "version": "1.0",
  "type": "task_delegate",
  "from": "node-a",
  "to": "node-b",
  "task_id": "uuid-v4",
  "timestamp": "2026-05-28T10:00:00+08:00",
  "payload": {
    "task_type": "chat",
    "instruction": "你好",
    "context": {},
    "session_id": "可选，用于多轮对话亲和"
  }
}
```

### task_result（任务结果）

```json
{
  "version": "1.0",
  "type": "task_result",
  "from": "node-b",
  "to": "node-a",
  "task_id": "uuid-v4",
  "status": "success",
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

| type | 说明 | 超时 | 走 AI |
|------|------|------|-------|
| `ping` | 存活检测 | 5s | ❌ |
| `chat` | 对话 | 120s | ✅ |
| `web_search` | 网络搜索 | 120s | ✅ |
| `coding` | 代码开发 | 120s | ✅ |
| `analysis` | 分析报告 | 120s | ✅ |

---

## API 端点

| Method | Path | 说明 |
|--------|------|------|
| GET | `/.well-known/agent-card.json` | Agent Card（Well-Known） |
| GET | `/a2a/.well-known/agent-card.json` | Agent Card 别名（兼容官方客户端） |
| POST | `/a2a` | 接收任务委托 |
| GET | `/a2a/{task_id}` | 查询任务状态 |
| POST | `/send` | 主动向对端发送任务 |
| GET | `/health` | 健康检查（包含 agentCardUrl） |
| GET | `/ready` | K8s 就绪探针 |
| GET | `/live` | K8s 存活探针 |
| GET | `/capabilities` | 支持的特性列表 |
| POST | `/report` | 接收对端工作汇报（reply_to 回调） |

---

## Docker 部署

```bash
# 节点 A
docker run -d \
  --name a2a-node-a \
  -p 8643:8643 \
  -e A2A_PORT=8643 \
  -e A2A_PEER_URL=http://peer:8644 \
  -e AI_PROVIDER_API_KEY=$AI_PROVIDER_API_KEY \
  -v $(pwd)/.env:/app/.env:ro \
  python:3.11-slim \
  bash -c "pip install fastapi uvicorn httpx && python a2a.py"
```

```yaml
# docker-compose.yml
services:
  node-a:
    image: python:3.11-slim
    ports:
      - "8643:8643"
    environment:
      A2A_PORT: 8643
      A2A_PEER_URL: http://node-b:8644
      AI_PROVIDER_API_KEY: ${AI_PROVIDER_API_KEY}
    volumes:
      - ./.env:/app/.env:ro
    working_dir: /app
    command: >-
      bash -c "pip install fastapi uvicorn httpx && python a2a.py"

  node-b:
    image: python:3.11-slim
    ports:
      - "8644:8644"
    environment:
      A2A_PORT: 8644
      A2A_NAME: node-b
      A2A_PEER_URL: http://node-a:8643
      AI_PROVIDER_API_KEY: ${AI_PROVIDER_API_KEY}
    volumes:
      - ./.env:/app/.env:ro
    working_dir: /app
    command: >-
      bash -c "pip install fastapi uvicorn httpx && python a2a.py"
```

---

## 目录结构

```
.
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
└── a2a.py              # 一体化主程序（接收 + 发送 + AI 调用）
```

---

## 版本历史

| 版本 | 日期 | 变更 |
|------|------|------|
| **v1.2.2** | 2026-05-28 | sync 参数（同步/异步双模式）；reply_to 回调字段；/report 端点（接收对端汇报） |
| **v1.2.1** | 2026-05-28 | 修复重复 app 实例导致所有路由 404 的 bug |
| **v1.2.0** | 2026-05-28 | **移除请求 Header 认证**（改从 env 读取）；修复 ping/instruction 空值 bug；增加 task_result 中继；扩大超时至 2048 |
| **v1.1.0** | 2026-05-26 | Agent Card + 客户端认证机制，符合官方 A2A Protocol 规范 |
| **v1.0.0** | 2026-05-26 | 一体化通用版本，A2A_PORT+A2A_PEER_URL 开箱即用，不分 client/server |
| v0.2.2.1 | 2026-05-14 | httpx 全局连接池 + 差异化超时 + 错误分类重试 |
| v0.2.1.0 | 2026-05-13 | 会话亲和（session_id 相同自动注入历史上下文） |
| v0.2.0.0 | 2026-05-13 | httpx AsyncClient 直调 AI，替代 subprocess；完整幂等存储 |
| v0.1.0.0 | 2026-05-12 | 初始版本（subprocess 方式） |

---

## 扩展方向

- [ ] WebSocket 双向推送
- [ ] 消息持久化（SQLite 表，支持重启恢复）
- [ ] mTLS 双向认证
- [ ] 多 Agent 路由（N > 2）
