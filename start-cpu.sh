#!/bin/bash

set -euo pipefail

cd /home/dwemer/parakeet-api-server/
if [ -s log.txt ]; then
    cp -- log.txt log.previous.txt
fi
: > log.txt
nohup /home/dwemer/parakeet-api-server/start_native.sh --cpu --precision int8 >>log.txt 2>&1 </dev/null &
