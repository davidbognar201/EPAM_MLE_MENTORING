-------------------------------------------------- VMs vs Containers --------------------------------------------------

As you've mentioned in the theory screening, I researched this topic more in depth and wrote a brief summary about the main differences of Virtual Machines and Containers.

So basically VMs and Containers are both isolated environments but in a very different way, by using VMs we isolate a whole machine with it's own allocated and virtualized RAM, disk space, network adapter and so on ( while the host machine sees it as a totally other, independent machine ), which is why it needs more resources while in Containers we pretty much isolate only proccesses and applications without the underlying hardware resources, this is why containers are more lightweight and more scalable for example in the case of multiple microservices.

The other big difference is what level the virtualization happens at. In case of VMs it is virtualized on a hardware level with a Type 1 or 2 hypervisor (it is responisble for all allocated components of the VM) while on the other hand, in case of containers the virtualization happens on a more higher level, at the level of host operating system by using 'namespaces' (responsible for managing the container's visibility and access to other processes on the system) and 'cgroups' (responisble for the dynamic resource allocation to the container).


-------------------------------------------------- Homework description --------------------------------------------------

I choosed to use a Gradient Boosting Regressor model from the "Advanced Regression" module of the Data Science training. 

At first I trained the model on my local machine and saved it as a .sav file with the pickle extension, in the container the testModel.py script will deserialize this .sav file and the pretrained model will be available for usage. 

Then I organised all the required files in different folders:
	/data -> The test and train files for the model
	/python_files -> training, testing and preprocessing scripts used during the assignment
	/trained_model -> the pretrained model as a .sav file
	/model_iteratios -> it will be created during the build, the main role of this folder is that the developer can export models in this files, because the trained_model is a read-only volume and the dev can't export anything to it

--- Dockerfile structure:

I tried to follow the Docker best practises where I could,
	- The layers are organized in an order that those layers are the first which are the least likely to change, so the building process is more effective this way because the more resoruce-intensive layers are cached and won't be built again during a rebuild
	- Sorted multi lines commad with \ for better readability 
	- There are no unnecessary packages and modules installed
	- If there would be anything unnecessary then I would exclude them with a .dockerignore file, but in this case there was no such files

In the Dockerfile I first started with the basic Linux image and then installed the most important dependencies:
	Python -> so the users will be able to run python scripts and develop 
	Pip -> so users can install more packages inside the container if needed
	Git -> so it is possible to use version control inside the container 


Then I installed some other dependencies (sklearn, numpy, pandas) from the requirements.txt, these are needed for the already existing scripts, but because pip is installed in the container, further packages can be also installed.

I defined two arguments 'USER_ID' and 'GROUP_ID', the values of these arguments will be loaded from the .env file by Docker Compose. The next two RUN layers will set up the correct group and user structure so the resources of the container won't be owned by the root user.

At the final steps I copied the python scripts to the working directory and created a new 'model_iterations' folder with mkdir.

--- Docker Compose structure:

The main reason why I created a compose file is that this way the container can be started with the least amount of additional console parameters for the volumes and user/group_id. The whole container can be set up with a 'docker-compose up' command.

In the 'build:' parameter of the compose file, I set the ARGs in the Dockerfile here under the 'args:' parameter. Because I declared an .env file, the ${GROUP_ID} and ${USER_ID} will be read from that file.

I choosed to create two read-only volumes, for the data and for the pretrained model. The reasoning behind this is that in a real-world scenario the base data is probably inmutable for us and by using volumes if the data changes (for example new records are added) then the container will immediately show these changes and we can work with the latest data without restarting the container.

At the end of the Compose file I added the following command "tail -f /dev/null", the purpose of this code is that the container won't stop after everything got executed in it. So this way the container will be up until we manually stop it.

