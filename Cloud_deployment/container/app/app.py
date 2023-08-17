# This is the file that implements a flask server to do inferences. It's the file that you will modify to
# implement the scoring for your own algorithm.

from __future__ import print_function

import io
import json
import os
import pickle
import signal
import sys
import traceback
import csv

import flask
import pandas as pd

import ScoringService as src




# The flask app for serving predictions
app = flask.Flask(__name__)


@app.route("/ping", methods=["GET"])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    health = src.ScoringService.get_model() is not None  

    status = 200 if health else 404
    return flask.Response(response="\n", status=status, mimetype="application/json")


@app.route("/invocations", methods=["POST"])
def transformation():
    """Do an inference on a single batch of data. In this sample server, we take data as CSV, convert
    it to a pandas data frame for internal use and then convert the predictions back to CSV (which really
    just means one prediction per line, since there's a single column.
    """
    data = None
    
    payload = flask.request.stream
    data = pd.read_json(payload, typ='frame', orient='split')

    print("Invoked with {} records".format(data.shape[0]))
    app.logger.info("Invoked with {} records".format(data.shape[0]))
    # Do the prediction
    predictions = src.ScoringService.predict(data)

    # Convert from numpy back to CSV
    out = io.StringIO()
    pd.DataFrame({"results": predictions}).to_csv(out, header=False, index=False)
    result = out.getvalue()

    return flask.Response(response=result, status=200, mimetype="text/csv")

if __name__ == "__main__":
    app.run(debug=True, port=8080)