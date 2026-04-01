#!/bin/bash
# Quick monitoring snapshot
cd /home/compute/trading

echo "=== Pipeline Monitor $(date -u '+%Y-%m-%d %H:%M UTC') ==="

# Process check
if kill -0 699245 2>/dev/null; then
    echo "Pipeline: ALIVE (PID 699245)"
else
    echo "Pipeline: DEAD!"
fi

# Heartbeat
HB=$(cat state_data/heartbeat.json 2>/dev/null)
echo "Heartbeat: $HB"

# Errors since last check
ERRORS=$(grep -c "ERROR\|FAIL\|Traceback" logs/pipeline_live.log 2>/dev/null)
echo "Error count in log: $ERRORS"

# Position
python3 -c "
import requests
r = requests.get('https://indexer.dydx.trade/v4/perpetualPositions?address=dydx1p2g0lgerv37jw5x3rgw64jh5eu30smclg58lvh&subaccountNumber=0&status=OPEN', timeout=10)
positions = r.json().get('positions', [])
r2 = requests.get('https://indexer.dydx.trade/v4/perpetualMarkets?ticker=BTC-USD', timeout=10)
oracle = float(r2.json().get('markets',{}).get('BTC-USD',{}).get('oraclePrice',0))
r3 = requests.get('https://indexer.dydx.trade/v4/addresses/dydx1p2g0lgerv37jw5x3rgw64jh5eu30smclg58lvh/subaccountNumber/0', timeout=10)
equity = r3.json().get('subaccount',{}).get('equity','?')
print(f'Oracle: \${oracle:,.2f} | Equity: \${equity}')
if positions:
    p = positions[0]
    entry = float(p.get('entryPrice',0))
    pnl = float(p.get('unrealizedPnl',0))
    pnl_pct = ((oracle - entry) / entry) * 100 if entry else 0
    print(f'Position: {p.get(\"side\")} {p.get(\"size\")} BTC @ \${entry:,.2f} | PnL: {pnl_pct:+.3f}% (\${pnl:+.4f})')
else:
    print('Position: FLAT (no open positions)')
" 2>/dev/null

# Last decision
python3 -c "
import json
with open('llm_agent/decision.json') as f:
    d = json.load(f)
print(f'Last decision: {d.get(\"direction\")} conf={d.get(\"confidence\",0):.2f}')
" 2>/dev/null

echo "==="
