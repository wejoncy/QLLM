#!/usr/bin/env bash
set -e

echo "=== Running flake8 (syntax errors & undefined names) ==="
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics

echo "=== Running flake8 (warnings) ==="
flake8 ./qllm/modeling/q_layers --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
flake8 ./qllm/quantization --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
flake8 ./qllm/utils --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

echo "All lint checks passed!"
