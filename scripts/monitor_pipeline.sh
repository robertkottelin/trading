#!/usr/bin/env bash
# Monitor pipeline health every 30 min for 12 hours, log to file
LOG="/home/compute/trading/logs/monitor_$(date +%Y%m%d_%H%M%S).log"
CHECKS=24  # 24 x 30min = 12 hours

for i in $(seq 1 $CHECKS); do
    echo "=== CHECK $i/$CHECKS ($(date -u '+%Y-%m-%d %H:%M UTC')) ===" >> "$LOG"
    
    # Is process alive?
    if pgrep -f "run_pipeline.py.*--no-testnet" > /dev/null; then
        echo "  Pipeline: RUNNING (PID $(pgrep -f 'run_pipeline.py.*--no-testnet'))" >> "$LOG"
    else
        echo "  Pipeline: DOWN — restarting..." >> "$LOG"
        cd /home/compute/trading
        nohup python run_pipeline.py --no-testnet --live --loop --interval 300 -v >> logs/pipeline_live.log 2>&1 &
        echo "  Restarted with PID $!" >> "$LOG"
    fi
    
    # Cycle count & errors
    CYCLES=$(grep -c "Pipeline run completed" /home/compute/trading/logs/pipeline_live.log 2>/dev/null || echo 0)
    ERRORS=$(grep -c "ERROR\|Traceback" /home/compute/trading/logs/pipeline_live.log 2>/dev/null || echo 0)
    echo "  Cycles: $CYCLES completed, Errors: $ERRORS" >> "$LOG"
    
    # Last decision
    grep "Grok decision:" /home/compute/trading/logs/pipeline_live.log | tail -1 >> "$LOG"
    
    # Heartbeat
    echo "  Heartbeat: $(cat /home/compute/trading/state_data/heartbeat.json 2>/dev/null)" >> "$LOG"
    echo "" >> "$LOG"
    
    [ "$i" -lt "$CHECKS" ] && sleep 1800
done
echo "=== MONITORING COMPLETE ===" >> "$LOG"
