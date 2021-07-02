from mlflow.pyfunc import scoring_server, multi_scoring_server
from mlflow import pyfunc

app = multi_scoring_server.init()
