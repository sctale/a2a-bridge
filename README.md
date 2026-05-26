# A2A Bridge — 通用双向 Agent 通信框架 v1.0.0

[![Version](https://img.shields.io/badge/version-v1.0.0-blue)](https://github.com/sctale/a2a-bridge/releases)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

> 让任意两个 AI Agent 通过 HTTP+JSON 互相通信，无需依赖第三方 A2A 库。
> **一体化开箱即用**：每个实例既是 Server 也是 Client，只配一个端口和对方地址即可。

---

## 核心特性

- **一体化设计**：不区分 client/server，每个节点既能接收也能主动发送任务
- **环境变量驱动**：所有配置通过 env 注入，不需要硬编码
- **事件循环永远不堵**：httpx AsyncClient + 线程池，所有阻塞代码异步化
- **幂等**：SQLite task_id 查重，24h TTL，failed 自动删除可重试
- **会话亲和**：session_id 相同则自动注入历史上下文，支持多轮对话
- **健康探针**：/health + /ready（K8s）+ /live + /capabilities
- **差异化超时**：ping 5s，其他 120s，429 指数退避重试

---

## 快速开始

### 1. 安装依赖

```bash
pip install fastapi uvicorn httpx
```

### 2. 启动节点 A（端口 8643）

```bash
A2A_PORT=8643 \
A2A_PEER_URL=http://localhost:8644 \
AI_PROVIDER_API_KEY=sk-xxx \
python a2a.py
```

### 3. 启动节点 B（端口 8644）

```bash
A2A_PORT=8644 \
A2A_PEER_URL=http://localhost:8643 \
AI_PROVIDER_API_KEY=sk-xxx \
python a2a.py
```

### 4. 验证

```bash
# A ← B：ping
curl -X POST http://localhost:8643/a2a \
  -H "Content-Type: application/json" \
  -d '{
    "version": "1.0",
    "type": "task_delegate",
    "from": "node-b",
    "to": "node-8643",
    "task_id": "ping-test",
    "payload": {"task_type": "ping", "instruction": ""}
  }'

# A → B：主动发送
curl -X POST http://localhost:8644/send \
  -H "Content-Type: application/json" \
  -d '{
    "task_type": "chat",
    "instruction": "你好，介绍一下你自己",
    "session_id": "sess-001"
  }'
```

---

## 配置说明

| 环境变量 | 默认值 | 说明 |
|---------|--------|------|
| `A2A_PORT` | `8643` | 监听端口 |
| `A2A_PEER_URL` | 空 | 对端地址（用于主动发送，不配则只收不发） |
| `A2A_NAME` | `node-{port}` | 节点名称，响应中标识身份 |
| `AI_PROVIDER_API_KEY` | 空 | AI API Key（处理收到的任务） |
| `AI_PROVIDER_BASE_URL` | `https://api.minimaxi.com/anthropic` | AI API 地址 |
| `A2A_MODEL` | `gpt-4o` | AI 模型 |
| `A2A_MAX_TOKENS` | `1024` | 最大 token 数 |
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
  "to": "node-8643",
  "task_id": "uuid-v4",
  "timestamp": "2026-05-26T10:00:00+08:00",
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
  "from": "node-8643",
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
| POST | `/a2a` | 接收任务委托 |
| GET | `/a2a/{task_id}` | 查询任务状态 |
| POST | `/send` | 主动向对端发送任务 |
| GET | `/health` | 健康检查 |
| GET | `/ready` | K8s 就绪探针 |
| GET | `/live` | K8s 存活探针 |
| GET | `/capabilities` | 支持的特性列表 |

---

## Docker 部署

```yaml
# docker-compose.yml
services:
  a2a-node:
    image: python:3.11-slim
    ports:
      - "8643:8643"
    volumes:
      - ./.env:/app/.env:ro
    working_dir: /app
    command: >
      bash -c "pip install fastapi uvicorn httpx &&
               A2A_PORT=8643
               A2A_PEER_URL=http://peer:8644
               A2A_NAME=my-agent
               python a2a.py"
```

---

## 目录结构

```
.
├── README.md
├── LICENSE
├── requirements.txt
└── a2a.py              # 一体化主程序（接收 + 发送 + AI 调用）
```

---

## 版本历史

| 版本 | 日期 | 变更 |
|------|------|------|
| **v1.0.0** | 2026-05-26 | **一体化通用版本** — 一个 py 文件，`A2A_PORT`+`A2A_PEER_URL` 开箱即用，不分 client/server；/send 主动发送端点；K8s 探针 |
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