@echo off

REM Check if output.csv exists, if not create an empty one
if not exist ".\python_implem\output.csv" (
    echo. > .\python_implem\output.csv
)

REM Start the Node.js Express.js server on port 3000
start /B node ".\python_implem\server\Express.js"
REM Wait for a moment to allow the Express.js server to start
timeout /T 2 >nul

cd ".\tsp-ol-app\"
start /B npm run start
REM Wait for a moment to allow the OL application server to start
timeout /T 2 >nul

cd "../"

REM Start the python script
start /B python ".\python_implem\EA.py" -i ".\python_implem\datasets\worldcities_10k.json" -o ".\python_implem\output.csv" -r ".\python_implem\results"
REM Wait for a moment to ensure output.csv is created (adjust the wait time as needed)
timeout /T 10 >nul


echo Press any key to stop the servers...
pause >nul

REM Terminate the Python and Node.js servers
taskkill /F /IM node.exe
taskkill /F /IM python.exe