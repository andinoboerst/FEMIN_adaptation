project_name := project

# mkfile_path := $(abspath $(lastword $(MAKEFILE_LIST)))
# current_dir := $(patsubst %/,%,$(dir $(mkfile_path)))

# build:
# 	docker build -t $(project_name)_image .

# run:
# 	docker run -it -p 8888:8888 -v $(current_dir)/$(project_name):/app --rm --name $(project_name) $(project_name)_image


# Variables
IMAGE_NAME = $(project_name)_image
CONTAINER_NAME = $(project_name)_container
mMAKEFILE_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
WORKSPACE_DIR := $(patsubst %/,%,$(dir $(mkfile_path)))
# WORKSPACE_DIR = $(shell pwd)

# Build the Docker image
build:
	docker build -t $(IMAGE_NAME) .devcontainer/

# Run the container and mount the current directory
run:
	docker run -dit --rm \
		--name $(CONTAINER_NAME) \
		-v $(WORKSPACE_DIR)/workspace:/workspace \
		$(IMAGE_NAME)

# Attach VS Code to the running container
attach:
	code --folder-uri vscode-remote://attached-container+$(shell docker inspect --format '{{.Id}}' $(CONTAINER_NAME))/workspace

# Start the development environment (build + run + attach)
dev: build run attach

# Stop and remove the container
clean:
	docker stop $(CONTAINER_NAME) || true
	docker rm $(CONTAINER_NAME) || true

# Remove the Docker image
prune:
	docker rmi $(IMAGE_NAME) || true

# Automatically clean up when the container is closed
auto-clean: dev
	@echo "Waiting for container to stop..."
	@docker wait $(CONTAINER_NAME)
	@echo "Container stopped. Cleaning up..."
	@make clean