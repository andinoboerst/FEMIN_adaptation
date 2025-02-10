FROM dolfinx/dolfinx:stable

RUN apt-get update && \
      apt-get -y install sudo

RUN apt-get install --fix-missing -y xvfb

RUN pip install matplotlib pyvista[jupyter] trame trame-vtk jupyter imageio

# Set the working directory to /workspace
WORKDIR /workspace
