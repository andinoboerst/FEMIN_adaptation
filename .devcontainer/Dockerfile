FROM --platform=linux/amd64 dolfinx/dolfinx:stable

RUN apt-get update && \
      apt-get -y install sudo

RUN apt-get install --fix-missing -y xvfb

RUN pip install --upgrade pip

RUN pip install matplotlib pyvista[jupyter] trame trame-vtk jupyter imageio imageio[ffmpeg] scikit-learn

# Set the working directory to /workspace
WORKDIR /workspace
