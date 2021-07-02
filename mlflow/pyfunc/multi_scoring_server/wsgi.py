import os
from mlflow.pyfunc import scoring_server, multi_scoring_server
from mlflow.pyfunc import load_model


app = multi_scoring_server.init()
