#!/usr/bin/env bash
# A2A Bridge 健康检查脚本
# 用法：./healthcheck.sh [PORT]  （默认 8643）
# 退出码：0=健康，1=不健康
#
# 可接入 cron / systemd 看门狗：
#   */5 * * * * /opt/data/github-a2a/scripts/healthcheck.sh 8643 || systemctl restart a2a-bridge

set -e
PORT="${1:-8643}"
URL="http://localhost:${PORT}/health"
TIMEOUT=3

# 1. /health HTTP 200 + status=ok
RESP=$(curl -s -m "$TIMEOUT" "$URL" 2>&1) || {
  echo "[FAIL] ${URL} 连接失败"
  exit 1
}

STATUS=$(echo "$RESP" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('status',''))" 2>/dev/null) || {
  echo "[FAIL] /health 响应不是合法 JSON: $RESP"
  exit 1
}

if [ "$STATUS" != "ok" ]; then
  echo "[FAIL] /health status=$STATUS (期望 ok): $RESP"
  exit 1
fi

NAME=$(echo "$RESP" | python3 -c "import json,sys; print(json.load(sys.stdin).get('name',''))" 2>/dev/null)
PEER=$(echo "$RESP" | python3 -c "import json,sys; print(json.load(sys.stdin).get('peer','null'))" 2>/dev/null)

echo "[OK] port=${PORT} name=${NAME} peer=${PEER}"
exit 0
