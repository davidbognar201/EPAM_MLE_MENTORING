I choosed to use Ubuntu during the Mentoring program as I would consider myself a Linux newbie, I've never used it in a professional environment yet, only during some of my university courses. 

During the installing process I pretty much followed the offical installing guide from the Docker site.

Installing Docker:
  > su (Logging in as a root user so I can run sudo commands)
  > sudo apt install gnome-terminal (prerequisity for Docker install)
  > sudo apt-get update
  > sudo apt-get install ca-certificates curl gnupg
  > sudo install -m 0755 -d /etc/apt/keyrings
  > curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
  > sudo chmod a+r /etc/apt/keyrings/docker.gpg
  > echo \
      "deb [arch="$(dpkg --print-architecture)" signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
      "$(. /etc/os-release && echo "$VERSION_CODENAME")" stable" | \
      sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
 > sudo apt-get update
 > sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

 
 After that I could run the "docker run hello-world" command only with root user, so I added my nonroot user to the Docker group:
 > sudo groupadd docker (but the Docker group already exists)
 > sudo gpasswd -a $USER docker ( adding all the other logged in nonrott users to the docker group)
 > newgrp docker (reevaluating the group memberships so the changes will apply to the group)
 > exit (after that the commands are run by my nonroot user)
 
 > docker run hello-world (Running the hello-world container, I am able to run it with my nonrott user so Docker seems to be installed        correctly)
 > docker compose version (The currently installed compose version is v2.18.1)
 > docker version (The currently installed docker version is 24.0.1)
