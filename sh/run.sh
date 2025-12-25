#!/bin/bash

# エラーが出たら停止
set -e

# プロジェクトルートに移動（このスクリプトが sh/ にある場合）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

echo "Starting Parallel Experiments..."
echo "Working directory: $(pwd)"

# ログディレクトリを作成
LOG_DIR="data/internal/exp0.05"
mkdir -p "$LOG_DIR"

# 4つの実験をバックグラウンド (&) で並列実行
# nohup を使ってターミナルを閉じても動き続けるようにする
# ログはそれぞれ別のファイルに出力する
# --output-dir オプションで統計ファイルもLOG_DIRに保存
nohup uv run python -u scripts/runrun.py --mode irl --output-dir "$LOG_DIR" > "$LOG_DIR/logs_irl.txt" 2>&1 &
PID1=$!
echo "Started Proposed (IRL) [PID: $PID1]"

nohup uv run python -u scripts/runrun.py --mode irl_no_second_term --output-dir "$LOG_DIR" > "$LOG_DIR/logs_irl_no_second_term.txt" 2>&1 &
PID2=$!
echo "Started Proposed (IRL, no 2nd term) [PID: $PID2]"

nohup uv run python scripts/runrun.py --mode softql --output-dir "$LOG_DIR" > "$LOG_DIR/logs_softql.txt" 2>&1 &
PID3=$!
echo "Started Oracle (SoftQL) [PID: $PID3]"

nohup uv run python scripts/runrun.py --mode softvi --output-dir "$LOG_DIR" > "$LOG_DIR/logs_softvi.txt" 2>&1 &
PID4=$!
echo "Started Oracle (SoftVI) [PID: $PID4]"

# PIDをファイルに保存（後で確認できるように）
echo "$PID1" > "$LOG_DIR/.pids_irl.txt"
echo "$PID2" > "$LOG_DIR/.pids_irl_no_second_term.txt"
echo "$PID3" > "$LOG_DIR/.pids_softql.txt"
echo "$PID4" > "$LOG_DIR/.pids_softvi.txt"

echo ""
echo "All experiments started in background!"
echo "You can safely close the terminal. Processes will continue running."
echo ""
echo "PID files saved: $LOG_DIR/.pids_irl.txt, $LOG_DIR/.pids_irl_no_second_term.txt, $LOG_DIR/.pids_softql.txt, $LOG_DIR/.pids_softvi.txt"
echo "To check progress: tail -f $LOG_DIR/logs_irl.txt (or logs_irl_no_second_term.txt, logs_softql.txt, logs_softvi.txt)"
echo "To check if still running: ps -p $PID1 $PID2 $PID3 $PID4"
echo ""
echo "Waiting for all experiments to finish..."
echo "(Press Ctrl+C to detach - processes will continue in background)"

# 全てのバックグラウンドプロセスが終わるのを待つ
# Ctrl+Cで中断しても、nohupで実行されているのでプロセスは続行される
trap 'echo ""; echo "Detached from processes. They will continue running in background."; exit 0' INT TERM

wait $PID1 $PID2 $PID3 $PID4

echo "All experiments finished!"

# 最後にまとめてプロット
echo "Generating plots..."
uv run python scripts/runrun.py --mode plot --output-dir "$LOG_DIR"

echo "Done! Check $LOG_DIR/comparison_curves.png"