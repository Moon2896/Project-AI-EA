#!/bin/bash

# Start the Python HTTP server on port 8000
# Assuming python3 is installed on your system
# Wait for a moment to allow the Python server to start
# Start the Node.js Express.js server on port 3000

node "./python_implem/server/Express.js" &
# Wait for a moment to allow the Express.js server to start
sleep 2

cd "./tsp-ol-app/"
npm run start &

sleep 2

cd "../"
# Open Express.js server to fetch data in Chrome
# xdg-open http://localhost:3000/data &

# # Open OL application in Chrome
# xdg-open http://127.0.0.1:5173/ &
# sleep 2

# Start the python script
# python3 ./python/EA.py  &
python3 "python_implem/EA.py" -i "python_implem/datasets/worldcities_10k.json" -o "python_implem/output.csv" -r "python_implem/results" &

# Wait for user input to stop the servers
read -p "Press [Enter] to stop the servers..."

# Terminate the Python and Node.js servers
pkill -f "node"
