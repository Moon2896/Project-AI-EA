# Evolutionary optimization assignment

This project is split between the evolutionary algorithm implementation and visualization tools made in node.js.
You will find the algorithm in the python_implem folder and the visualisation tool in the tsp-ol-app folder.

## Setup
You will need python3 with version at least 3.9, aswell as node.js at least 16 in order to install the dependencies for this project.

Let's install the dependencies.

While we recommend using vritual environments, as they will ensured an isolated environment, they are not mandatory.

1. Create a new virtual environment:
>python3 -m venv ./python_implem/venv>
2. Activate the environment
    ### Windows
    >./python_implem/venv/Scripts/Activate.ps1
    ### Linux
    >source ./python_implem/venv/bin/activate

3. Install python dependencies
>pip install -r ./python_implem/requirements.txt

4. Install python server dependencies
>cd ./python_implem/server
>npm i
>cd ../..

5. Install visualisation dependencies
>cd tsp-ol-app
>npm i
>cd ..

## Execution
You can either start the whole thing with default parameters by using the start-servers scripts, or first start the servers and then execute the algorithm if you want to modify the parameters:

### Manual start:
Make sure you have the './python_implem/output.csv' file exists. You can create it empty if needed.

1. Go to ./python_implem/server and run:
>npm start

2. Go to ./tsp-ol-app and run:
>npm start

3. From the root of the repository, run:
python3 "./python_implem/EA.py" and have a look at the parameters !