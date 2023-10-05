#!/bin/bash

# Start the python script
python3 "python_implem/EA.py" -i "python_implem/datasets/worldcities_10k.json" -o "python_implem/output.csv" -r "python_implem/results" &
# Save the PID of the Python process
PYTHON_PID=$!

sleep 2
# Start the Node.js Express.js server on port 3000
node "./python_implem/server/Express.js" &
# Save the PID of the process
NODE_PID=$!
# Wait for a moment to allow the Express.js server to start
sleep 2

cd "./tsp-ol-app/"
npm run start &
# Save the PID of the npm process
NPM_PID=$!
sleep 2
cd ".."

# Wait for user input to stop the servers
read -p "Press [Enter] to stop the servers..."

# Terminate the Python and Node.js servers using their PIDs
kill $NODE_PID
kill $NPM_PID
kill $PYTHON_PID
