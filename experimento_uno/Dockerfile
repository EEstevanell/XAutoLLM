# Use the PyTorch base image
FROM nvidia/cuda:12.2.0-runtime-ubuntu20.04

# Pass in user/group info as build args
ARG USER_ID
ARG GROUP_ID

# # Install pip (if not already installed) and upgrade it
# RUN apt-get update -y && \
#     apt-get install -y python3-pip && \
#     python3 -m pip install --upgrade pip

# # Install everything in one go to avoid potential environment confusion
# RUN python3 -m pip install --no-cache-dir \
#     torch \
#     torchaudio \
#     torchvision \
#     nltk \
#     pandas \
#     numpy \
#     tqdm \
#     torch torchvision torchaudio \
#     transformers \
#     stanza \
#     wn \
#     svgling \
#     datasets \
#     regex    \
#     scikit-learn

# RUN python3 -m pip install --upgrade --no-cache-dir transformers


RUN apt-get update -y
RUN apt-get install pip -y

RUN pip install pandas
RUN pip install numpy
RUN pip install tqdm
RUN pip install scispacy
RUN pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_core_sci_sm-0.5.1.tar.gz
RUN pip install networkx[default]
RUN pip install torch 
RUN pip install torchvision
RUN pip install torchaudio
RUN pip install transformers
RUN pip install black
RUN pip install segtok
RUN pip install scikit-learn
RUN pip install datasets
RUN pip install accelerate

# Create a non-root user
RUN groupadd -g $GROUP_ID usergroup && \
    useradd -m -u $USER_ID -g $GROUP_ID user && \
    mkdir -p /workspace && \
    chown -R user:usergroup /workspace

# Switch to this new user and set the working directory
USER user
WORKDIR /workspace

# By default, run your experiment script
#CMD ["nohup", "/workspace/experiments/kan_heads/Code/polynomials/experiment_setup.sh"]
ENTRYPOINT ["/bin/bash", "-c", "/workspace/experiments/kan_heads/Code/polynomials/experiment_setup.sh"]