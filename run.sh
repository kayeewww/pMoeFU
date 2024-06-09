#!/bin/bash
set -e
exec > ./log/ssh-client$(date +%Y-%m-%d_%H-%M-%S).log 2>&1  # Redirect stdout and stderr to log.txt
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/

python server.py &#--toy  &
sleep 30  # Sleep for 10s to give the server enough time to start and download the dataset
CLIENT_BASE_PORT=8081

for i in `seq 0 9`; do
  PORT=$((CLIENT_BASE_PORT + i))
  echo "Starting client $i on port $PORT"
  python client.py --port=$PORT --client-id=${i} &#--toy &
done

# Enable CTRL+C to stop all background processes
trap "trap - SIGTERM and kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait
