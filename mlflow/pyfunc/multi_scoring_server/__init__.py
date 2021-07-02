from collections import OrderedDict
import flask
import json
import logging
import numpy as np
import pandas as pd
import sys
import traceback

# NB: We need to be careful what we import form mlflow here. Scoring server is used from within
# model's conda environment. The version of mlflow doing the serving (outside) and the version of
# mlflow in the model's conda environment (inside) can differ. We should therefore keep mlflow
# dependencies to the minimum here.
# ALl of the mlfow dependencies below need to be backwards compatible.
from mlflow.exceptions import MlflowException
from mlflow.pyfunc.multi_scoring_server.lib import get_desired_model
from mlflow.pyfunc.scoring_server import parse_csv_input, infer_and_parse_json_input, parse_json_input, \
    parse_split_oriented_json_input_to_numpy, _handle_serving_error, predictions_to_json
from mlflow.types import Schema
from mlflow.utils import reraise
from mlflow.utils.proto_json_utils import (
    NumpyEncoder,
    _dataframe_from_json,
    _get_jsonable_obj,
    parse_tf_serving_input,
)

try:
    from mlflow.pyfunc import load_model, PyFuncModel
except ImportError:
    from mlflow.pyfunc import load_pyfunc as load_model
from mlflow.protos.databricks_pb2 import MALFORMED_REQUEST, BAD_REQUEST
from mlflow.server.handlers import catch_mlflow_exception

try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO

_SERVER_MODEL_PATH = "__pyfunc_model_path__"

CONTENT_TYPE_CSV = "text/csv"
CONTENT_TYPE_JSON = "application/json"
CONTENT_TYPE_JSON_RECORDS_ORIENTED = "application/json; format=pandas-records"
CONTENT_TYPE_JSON_SPLIT_ORIENTED = "application/json; format=pandas-split"
CONTENT_TYPE_JSON_SPLIT_NUMPY = "application/json-numpy-split"

CONTENT_TYPES = [
    CONTENT_TYPE_CSV,
    CONTENT_TYPE_JSON,
    CONTENT_TYPE_JSON_RECORDS_ORIENTED,
    CONTENT_TYPE_JSON_SPLIT_ORIENTED,
    CONTENT_TYPE_JSON_SPLIT_NUMPY,
]

_logger = logging.getLogger(__name__)


def init():

    """
    Initialize the server. Loads pyfunc model from the path.
    """
    app = flask.Flask(__name__)
    #input_schema = model.metadata.get_input_schema()

    # @app.route("/ping", methods=["GET"])
    # def ping():  # pylint: disable=unused-variable
    #     """
    #     Determine if the container is working and healthy.
    #     We declare it healthy if we can load the model successfully.
    #     """
    #     health = model is not None
    #     status = 200 if health else 404
    #     return flask.Response(response="\n", status=status, mimetype="application/json")

    @app.route("/invocations/<model_uri>/<version>", methods=["POST"])
    @catch_mlflow_exception
    def transformation(model_uri, version):  # pylint: disable=unused-variable
        """
        Do an inference on a single batch of data. In this sample server,
        we take data as CSV or json, convert it to a Pandas DataFrame or Numpy,
        generate predictions and convert them back to json.
        """
        print("IN /invocations/<model_uri>/<version>")
        model = get_desired_model("models:/{uri}/{version}".format(uri=model_uri, version=version))
        input_schema = model.metadata.get_input_schema()

        # Convert from CSV to pandas
        if flask.request.content_type == CONTENT_TYPE_CSV:
            data = flask.request.data.decode("utf-8")
            csv_input = StringIO(data)
            data = parse_csv_input(csv_input=csv_input)
        elif flask.request.content_type == CONTENT_TYPE_JSON:
            json_str = flask.request.data.decode("utf-8")
            data = infer_and_parse_json_input(json_str, input_schema)
        elif flask.request.content_type == CONTENT_TYPE_JSON_SPLIT_ORIENTED:
            data = parse_json_input(
                json_input=StringIO(flask.request.data.decode("utf-8")),
                orient="split",
                schema=input_schema,
            )
        elif flask.request.content_type == CONTENT_TYPE_JSON_RECORDS_ORIENTED:
            data = parse_json_input(
                json_input=StringIO(flask.request.data.decode("utf-8")),
                orient="records",
                schema=input_schema,
            )
        elif flask.request.content_type == CONTENT_TYPE_JSON_SPLIT_NUMPY:
            data = parse_split_oriented_json_input_to_numpy(flask.request.data.decode("utf-8"))
        else:
            return flask.Response(
                response=(
                    "This predictor only supports the following content types,"
                    " {supported_content_types}. Got '{received_content_type}'.".format(
                        supported_content_types=CONTENT_TYPES,
                        received_content_type=flask.request.content_type,
                    )
                ),
                status=415,
                mimetype="text/plain",
            )

        # Do the prediction

        try:
            raw_predictions = model.predict(data)
        except MlflowException as e:
            _handle_serving_error(
                error_message=e.message, error_code=BAD_REQUEST, include_traceback=False
            )
        except Exception:
            _handle_serving_error(
                error_message=(
                    "Encountered an unexpected error while evaluating the model. Verify"
                    " that the serialized input Dataframe is compatible with the model for"
                    " inference."
                ),
                error_code=BAD_REQUEST,
            )
        result = StringIO()
        predictions_to_json(raw_predictions, result)
        return flask.Response(response=result.getvalue(), status=200, mimetype="application/json")

    return app