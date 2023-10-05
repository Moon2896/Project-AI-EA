@echo off
REM Start the Node.js Express.js server on port 3000
start /B node ".\python_implem\server\Express.js"
REM Wait for a moment to allow the Express.js server to start
timeout /T 2 >nul

cd ".\tsp-ol-app\"
start /B npm run start
REM Wait for a moment to allow the OL application server to start
timeout /T 2 >nul

REM Open Express.js server to fetch data in Chrome
REM start http://localhost:3000/data

REM Open OL application in Chrome
REM start http://127.0.0.1:5173/
REM timeout /T 2 >nul

REM Start the python script
start /B python ".\python_implem\EA.py" -i ".\python_implem\datasets\worldcities_10k.json" -o ".\python_implem\output.csv" -r ".\python_implem\results"

echo Press any key to stop the servers...
pause >nul

REM Terminate the Python and Node.js servers
taskkill /F /IM node.exe
