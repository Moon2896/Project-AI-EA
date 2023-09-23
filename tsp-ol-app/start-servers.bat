@echo off

@REM Start the Python HTTP server on port 8000
@REM Assuming python is in your system PATH
@REM Replace "full\path\to\python.exe" with the actual path to your Python executable if needed
@REM start "Python HTTP Server" "full\path\to\python.exe" -m http.server 8000

@REM Wait for a moment to allow the Python server to start
timeout /t 1 /nobreak

@REM Start the Node.js Express.js server on port 3000
cd "../python_implem/"
start "Node.js Express Server" npm start

@REM Wait for a moment to allow the Express.js server to start
timeout /t 1 /nobreak

@REM Wait for a moment to allow the Express.js server to start
timeout /t 1 /nobreak

@REM Express.js server to fetch data
start chrome http://localhost:3000/data

@REM OL application
start chrome http://127.0.0.1:5173/

@REM Start the python script
cd "../python_implem/python/"
start "Python Script" python ./EA.py

@REM Wait for user input to stop the servers (you can press Ctrl+C to stop them)
pause

@REM Terminate the Python and Node.js servers when the user presses Enter
taskkill /f /im "python.exe"
taskkill /f /im "node.exe"