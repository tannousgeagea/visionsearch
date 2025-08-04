FROM pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime

# Maintainer instructions has been deprecated, instead use LABEL
LABEL maintainer="tannous.geagea@wasteant.com"

# Versionining as "b-beta, a-alpha, rc - release candidate"
LABEL com.wasteant.version="1.1b1"

# [CHECK] Whether it is convenient to use the local user values or create ENV variables, or run everyhting with root
ARG user=appuser
ARG userid=1000
ARG group=appuser
ARG groupid=1000

# Install other necessary packages and dependencies
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -q -y --no-install-recommends \
    apt-utils \
	vim \
	iputils-ping \
	net-tools \
	netcat \
    curl \
    lsb-release \
    wget \
    zip \
    sudo \
    && rm -rf /var/lib/apt/lists/*

# Install Python + dependencies
RUN apt-get update && apt-get install -y \
    python3.10 python3.10-dev python3-pip python3-venv \
    ffmpeg libsm6 libxext6 git curl \
    && rm -rf /var/lib/apt/lists/*

# Set python3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

RUN pip3 install supervisor
RUN pip3 install fastapi
RUN pip3 install uvicorn[standard]
RUN pip3 install gunicorn
RUN pip3 install python-multipart
RUN pip3 install pydantic
RUN pip3 install django==4.2
RUN pip3 install django-unfold
RUN pip3 install django-storages[azure]
RUN pip3 install psycopg2-binary

# Vision Search
RUN pip3 install faiss-cpu
RUN pip3 install clip-anytorch
RUN pip3 install Pillow

# Perception Encoder PE
COPY ./visionsearch/common_utils/perception_models/requirements.txt /home/appuser/src/visionsearch/common_utils/perception_models/requirements.txt
RUN pip3 install --no-cache-dir -r /home/appuser/src/visionsearch/common_utils/perception_models/requirements.txt

# Gemini
RUN pip3 install -q -U google-genai

# upgrade everything
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get upgrade -q -y \
   && rm -rf /var/lib/apt/lists/*

# # Set up users and groups
RUN addgroup --gid $groupid $group && \
	adduser --uid $userid --gid $groupid --disabled-password --gecos '' --shell /bin/bash $user && \
	echo "$user ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers.d/$user && \
	chmod 0440 /etc/sudoers.d/$user

RUN mkdir -p /home/$user/src

RUN /bin/bash -c "chown -R $user:$user /home/$user/"
RUN /bin/bash -c "chown -R $user:$user /media"

# Create directory for Supervisor logs
RUN mkdir -p /var/log/supervisor && \
    chmod -R 755 /var/log/supervisor
	
COPY . /home/${user}/src
COPY ./supervisord.conf /etc/supervisord.conf
COPY ./entrypoint.sh /home/.
RUN /bin/bash -c "chown -R $user:$user /home/$user/"
RUN /bin/bash -c "chown $user:$user /home/entrypoint.sh"

ENTRYPOINT /bin/bash -c ". /home/entrypoint.sh"