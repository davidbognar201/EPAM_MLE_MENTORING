I used the first classification module's task for this project, the goal of the model is to classify thyroid positive/negative people based on clinical data.

I created 3 containerized services during the task:

	1., - Flask-server: A basic flask application which runs on 127.0.0.1:5000 where users can get prediction for a single observation. After the form is filled with valid data and submitted, the application loads the same encoder and scaler which was used during the training and preproccesses the data, then writes the predicted label with a probability score to the webpage.


	2., - Batch-server: A basic FastAPI application running on 127.0.0.1:8000, it can get POST requests on the '/predict-batch' URL. Expects an 'Input' object in JSON format, this object is pre-defined in my package's 'deployment_utils.src.schemas' file, it is basically a Python object which represents the raw data. 
		- After the request body is processed, it creates and instance from the DataPreprocess class, which is also in my package. This class contains all the function which are required for the preprocessing and predicting part.
		- Finally, it send back the predicted labels as a JSON object.
		- On startup it runs the defined unit tests in the 'deployment_package' package.


	3., - Client: This container's only job is to send POST requests to the batch-server in a given interval (at the moment it's 1 minute for the sake of easier testing).
		- The scheduling is made by using a cron job, which runs the test_request.py script every minute and the .py script produces a log_file in '/client/src/logs/request_log.txt', which contains the datetime of every run.

The source code for the custom package is in the following repository: https://github.com/davidbognar201/Model_deployment_package

Reproducing the HW:
    1, Run 'docker-compose up' int the Model_deployment folder.
    2, After everything is ready, the Flask app is available at 127.0.0.1:5000
    3, In the client/src/data folder a 'dataset_w_labels.csv' file will appear, this is the data file which contains the predicted labels for the sent data,
    4, In the client/src/logs folder the 'request_log.txt' will keep track of every run.

