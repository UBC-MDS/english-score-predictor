FROM quay.io/jupyter/minimal-notebook:2023-11-21

WORKDIR /home/jovyan

COPY environment.yml .

RUN conda env update -n base -f environment.yml
