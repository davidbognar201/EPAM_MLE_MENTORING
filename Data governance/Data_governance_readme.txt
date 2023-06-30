For this homework I used the NLP Classification task from the Data Science Training. I'm currently learning AWS so I wanted to experiment with it, so I choosed to store the data in an S3 bucket for this task.

--------- AWS setup ---------
	- I created a S3 bucket called 'epam-mle-mentoring-bucket' and a IAM user, who has only a custom policy, which gives the following permissions: ("s3:ListBucket", "s3:PutObject", "s3:GetObject").
	- I generated an acces key to this user, which will be needed during the dvc operations to access the remote storage.
	- The keys ( probably not the best idea to share it in a public repository but I will deactivate the key after the homework is graded and it also doesn't give much permission to my AWS so I think it will be good enough for this homework )
		- Access Key ID: AKIASMPFQSV2QDQKJF5Y
		- Secret access key: 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

--------- Folder structure ---------	
	- /src -> it contains all the necessary .py scripts to run the pipelines
	- /data -> the .csv files are pulled in this folder by dvc pull and repro


--------- DVC setup ---------
	- I created 2 pipelines (preprocess and traintestModel)
		- 'preprocess' just does the basic preprocessing tasks for an NLP model and saves the preprocessed data as a .csv file (this file will be a dependency for the nex pipeline, so it cannot run without it)
		- 'traintestModel' just splits the preprocessed data and trains a Stochastic Gradient Descent classifier on it, then saves the metrics into a .json file, so we can compare the metrics of different versions.

--------- Guide for reproducing the HW ---------
	- In the requirements.txt there are a few packages which are required by the scripts to run properly
	- Before pulling the data from dvc or running the pipeline, it is required to give the above AWS credentials in the CLI (I used the '$ aws configure' command, copied the two keys and left the two other options empty and it worked on both of my Linux and Windows machines)

