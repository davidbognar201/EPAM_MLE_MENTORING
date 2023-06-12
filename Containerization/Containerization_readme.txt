I choosed to use a Gradient Boosting Regressor model from the "Advanced Regression" module of the Data Science training. 

At first I trained the model on my local machine and saved it as a .sav file with the pickle extension, in the container the code will deserialize this .sav file and the pretrained model will be available for usage. 


Then I organised all the required files in different folders:
	/data -> The test and train files for the model
	/python_files -> training, testing and preprocessing scripts used during the assignment
	/trained_model -> the pretrained model as a .sav file

In the Dockerfile I first started with the basic Linux image and then installed the most important dependencies:
	Python -> so the users will be able to run python scripts and develop 
	Pip -> so users can install more packages inside the container if needed
	Git -> so it is possible to use version control inside the container 

After this layer I installed some other dependencies (sklearn, numpy, pandas) from the requirements.txt, these are needed for the already existing scripts. Finally, set the working directory (/model) and copied the pretrained model and scripts in it.
