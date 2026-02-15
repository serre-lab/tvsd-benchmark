IMAGE_NAME ?= tvsd-benchmark
IMAGE_TAG ?= latest

DOCKER_IMAGE := $(IMAGE_NAME):$(IMAGE_TAG)

.PHONY: help build test test-local shell cleanlogs cleantimm

help:
	@echo "Available targets:"
	@echo "  make build      Build Docker image ($(DOCKER_IMAGE))"
	@echo "  make test       Run unit tests in Docker"
	@echo "  make test-local Run unit tests locally"
	@echo "  make shell      Open an interactive shell in the Docker image"
	@echo "  make cleanlogs  Remove logs/*"
	@echo "  make cleantimm  Remove configs/timm/*"

build:
	docker build -t $(DOCKER_IMAGE) .

test:
	docker run --rm $(DOCKER_IMAGE) pytest -q tests

test-local:
	pytest -q tests

shell:
	docker run --rm -it $(DOCKER_IMAGE) /bin/bash

# cleanlogs: get rid of contents of logs/
cleanlogs:
	rm -rf logs/*
	@echo "Logs directory cleaned"

# cleantimm: get rid of contents of configs/timm/
cleantimm:
	rm -rf configs/timm/*
	@echo "Timm configs directory cleaned"
