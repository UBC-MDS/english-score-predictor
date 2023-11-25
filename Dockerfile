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