#!/bin/bash

# Start the Python HTTP server on port 8000
# Assuming python3 is installed on your system
pkill -f "python3 -m http.server"
pkill -f "node"

python3 -m http.server 8000 &

# Wait for a moment to allow the Python server to start
sleep 1

# Start the Node.js Express.js server on port 3000
cd "../python_implem/"
npm run start &

# Wait for a moment to allow the Express.js server to start
sleep 2

cd "../tsp-ol-app/"
npm start &

# Open Express.js server to fetch data in Chrome
# xdg-open http://localhost:3000/data &

# # Open OL application in Chrome
# xdg-open http://127.0.0.1:5173/ &
# sleep 2

# Start the python script
# python3 ./python/EA.py  &

# Wait for user input to stop the servers
read -p "Press [Enter] to stop the servers..."

# Terminate the Python and Node.js servers
pkill -f "python3 -m http.server"
pkill -f "node"
