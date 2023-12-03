FROM quay.io/jupyter/minimal-notebook:2023-11-21

# docker build -t 522_project .
# docker run --rm -p 8888:8888 -v $(pwd):/home/jovyan/work 522_project

WORKDIR /home/jovyan

COPY environment.yml .

RUN conda env update --file environment.yml

# EXPOSE 8888