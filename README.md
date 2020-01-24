# fastAPI ML quickstart

## Project setup
1. Create the virtual environment.
```
virtualenv /path/to/venv --python=/path/to/python3
```
You can find out the path to your `python3` interpreter with the command `which python3`.
2. Activate the environment and install dependencies.
```
source /path/to/venv/bin/activate
pip install -r requirements.txt
```

## Deployment with Docker
1. Build the Docker image
```
docker build --file Dockerfile --tag fastapi-ml-quickstart .
```

Entering into the Docker image
```
docker run -it --entrypoint /bin/bash fastapi-ml-quickstart
```

