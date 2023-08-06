FROM ubuntu

RUN apt-get update && \ 
            apt-get install -y python3 python3-pip && \
            apt-get -y install -qq --force-yes cron && \
            apt-get -y install nano && \
            apt-get -y install tofrodos

COPY requirements.txt .
RUN pip3 install -r requirements.txt

# Add cron_job file in the cron directory
ADD /cron/cron_job /etc/cron.d/cron_job

RUN fromdos /etc/cron.d/cron_job
# Give execution rights on the cron job
RUN chmod 0644 /etc/cron.d/cron_job

# Apply cron job
RUN crontab /etc/cron.d/cron_job

# Create the log file
RUN touch /var/log/cron.log

