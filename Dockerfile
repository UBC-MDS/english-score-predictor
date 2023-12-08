FROM quay.io/jupyter/minimal-notebook:2023-11-21

COPY environment.yml .

RUN conda env update -n base -f environment.yml

RUN conda install -y make
