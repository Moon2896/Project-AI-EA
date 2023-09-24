@echo off

rem Start the Python HTTP server on port 8000
start python -m http.server 8000

rem Wait for a moment to allow the Python server to start
timeout /t 2 /nobreak

rem Start the Node.js Express.js server on port 3000
start node Express.js

rem Wait for a moment to allow the Express.js server to start
timeout /t 2 /nobreak

@REM rem Run your Pixi.js application (you can replace this with your actual command)
@REM rem Example: npm start

rem Optionally, you can open a web browser to view the Pixi.js application
start chrome http://localhost:3000

rem Start the python script
start python -m ./python/EA.py

rem Wait for user input to stop the servers (you can press Ctrl+C to stop them)
pause

rem Terminate the Python and Node.js servers when the user presses Enter
taskkill /f /im "python.exe"
taskkill /f /im "node.exe"
