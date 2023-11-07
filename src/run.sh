#!/bin/bash

# ------------------------------------------------- #
# Author: xietao                					#
# Repo: https://github.com/xietao02/Federated-NCF/  #
# ------------------------------------------------- #


set -e

# activate conda
source /PATH/TO/YOUR/anaconda3/bin/activate YOUR_ENV_NAME

# Starting server
# python server.py &

sleep 3  # Sleep for 3s to give the server enough time to start

# Determine the value of 'seq' as needed.
for i in $(seq 2 10); do
    echo "Starting client $i"
    python client.py --cid $i &
    sleep 0.5
done

# Enable CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM

# Wait for all background processes to complete
wait