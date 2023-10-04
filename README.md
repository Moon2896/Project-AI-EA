# Evolutionary optimization assignment

This project is split between the evolutionary algorithm implementation and visualization tools made in node.js.
You will find the algorithm in the python_implem folder and the visualisation tool in the tsp-ol-app folder.

## Setup
You will need python3 with version at least 3.9, aswell as node.js at least 16 in order to install the dependencies for this project.

Let's install the dependencies.

While we recommend using vritual environments, as they will ensured an isolated environment, they are not mandatory.

Go into the python_implem folder:
``` cd python_implem ```

### Dependencies for the evolutionary algorithm

#### Creating a virtual environment (requires the venv package):
    ``` python3 -m venv venv```

2. Activate it:
On windows:
``` ./venv/Scripts/Activate.ps1```

On Linux:
``` source ./venv/Scripts/activate ```

#### Install dependencies
``` pip install -r requirements.txt ```

### Dependencies for the evolutionary algorithm server:
This server is used to serve the result file so that it can be accessed by our visualisation app later.

#### Install dependencies

```npm i```

### Dependencies for the visualization app

Go to the visualisation app folder

```cd ../tsp-ol-app```

#### Install dependencies

```npm i```

## Running the project

### On windows
Run start-server.bat file

### On linux
Run start-server.sh file (might need to chmod +x it depending on distribution)