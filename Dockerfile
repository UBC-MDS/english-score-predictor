FROM quay.io/jupyter/minimal-notebook:2023-11-21

# docker build -t 522_project .
# docker run --rm -p 8888:8888 -v $(pwd):/home/jovyan/work 522_project

RUN conda install -y numpy=1.26.* \
    pandas=2.1.* \
    scikit-learn=1.3.* \
    ipykernel=6.26.* \
    altair=5.1.* \
    matplotlib=3.8.* \
    scipy=1.11.* \
    jupyterlab=4.0.* \
    jupyterlab-git=0.41.*

RUN pip install "vegafusion[embed]"

EXPOSE 8888
# Got from https://pythonspeed.com/articles/activate-conda-dockerfile/
# COPY environment.yml .
# RUN conda env create -f environment.yml

# # Make RUN commands use the new environment:
# SHELL ["conda", "run", "-n", "522", "/bin/bash", "-c"]

# # Some code below is taken from: 
# # https://ubc-dsci.github.io/reproducible-and-trustworthy-workflows-for-data-science/materials/lectures/05-containerization.html#how-do-we-specify-a-container-image

# # Create working directory for mounting volumes
# RUN mkdir -p /opt/notebooks

# # Make port 8888 available for JupyterLab
# EXPOSE 8888

# # Copy JupyterLab start-up script into container
# COPY start-notebook.sh /usr/local/bin/

# # Change permission of startup script and execute it
# RUN chmod +x /usr/local/bin/start-notebook.sh
# ENTRYPOINT ["/usr/local/bin/start-notebook.sh"]

# # Switch to staring in directory where volumes will be mounted
# WORKDIR "/opt/notebooks"