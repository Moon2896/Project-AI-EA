#!/bin/bash

# Start the Python HTTP server on port 8000
python3 -m http.server 8000 &

# Wait for a moment to allow the Python server to start
sleep 2

# Start the Node.js Express.js server on port 3000
node Express.js &

# Wait for a moment to allow the Express.js server to start
sleep 2

# Start the python script
python3 -m ./python/EA.py &

# Wait for user input to stop the servers
read -p "Press [Enter] to stop the servers..."

# Terminate the Python and Node.js servers when the user presses Enter
pkill -f "python3 -m http.server"
pkill -f "node Express.js"

