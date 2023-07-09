For this homework I used the NLP Classification task from the Data Science Training. I'm currently learning AWS so I wanted to experiment with it, so I choosed to store the data in an S3 bucket for this task.

--------- AWS setup ---------
	- I created a S3 bucket called 'epam-mle-mentoring-bucket' and a IAM user, who has only a custom policy, which gives the following permissions: ("s3:ListBucket", "s3:PutObject", "s3:GetObject").
	- I generated an acces key to this user, which will be needed during the dvc operations to access the remote storage.

--------- Folder structure ---------	
	- /src -> it contains all the necessary .py scripts to run the pipelines
	- /data -> the .csv files are pulled in this folder by dvc pull and repro


--------- DVC and Docker setup ---------
	- I created 2 pipelines (preprocess and traintestModel)
		- 'preprocess' just does the basic preprocessing tasks for an NLP model and saves the preprocessed data as a .csv file (this file will be a dependency for the nex pipeline, so it cannot run without it)
		- 'traintestModel' just splits the preprocessed data and trains a Stochastic Gradient Descent classifier on it, then saves the metrics into a .json file, so we can compare the metrics of different versions.
	- The docker structure is basically the same as the one I used during the previous homework (without the volumes)
	- When starting the container the following codes will be executed:
		- bash -c "aws configure set aws_access_key_id ${AWS_KEY} # Using the AWS keys from the .env file in the container environment 
      && aws configure set aws_secret_access_key ${AWS_SECRET_KEY}	
      && aws sts get-caller-identity # Verifying that the AWS user is correctly configured
      && git init	# Initiating a version control in the container because DVC can't run without it
      && dvc pull # Pulling the data from the AWS S3 bucket to the container
      && dvc repro --force # Running the preprocessing and training pipelines
      && dvc metrics show # Will print the actual metrics of the training 
      && tail -f /dev/null" # The container will keep running until it's shut down manually 

--------- Guide for reproducing the HW ---------
	- Before setting up the container, the AWS keys must be copied in the .env file
	- After the keys are set, the container can be set up completely with a 'docker-compose up' command and it will run every command which is needed in the task
