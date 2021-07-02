from mlflow import pyfunc
from mlflow.tracking.artifact_utils import _download_artifact_from_uri


def get_desired_model(model_uri):
    # local_path = _download_artifact_from_uri(model_uri)
    model = pyfunc.load_pyfunc(model_uri)

    return model