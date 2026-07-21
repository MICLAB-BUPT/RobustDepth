PROJECT ?= uamtd
VERSION ?= latest
DOCKER_IMAGE ?= $(PROJECT)/ubuntu20.04-cuda11.3-pytorch1.13:${VERSION}

# Parent directory that contains your nuScenes dataset (the repo expects
# `../nuscenes` to point at it, e.g. symlink ../nuscenes -> $(DATAROOT_VOL)/nuscenes).
DATAROOT_VOL ?= <PATH_TO_NUSCENES_PARENT>

docker-build:
	docker build -f Dockerfile -t ${DOCKER_IMAGE} .

# Interactive shell inside the container (code mounted at /workspace).
docker-start:
	docker run --gpus all --rm -it \
		--shm-size=32g \
		-v ${PWD}:/workspace \
		-v ${DATAROOT_VOL}:/data \
		-w /workspace \
		${DOCKER_IMAGE} bash

# Train: make docker-train-<config-name-without-.yaml>
#   e.g. make docker-train-train_uncdistill_strongteacher
docker-train-%:
	docker run --gpus all --rm \
		-v ${PWD}:/workspace \
		-v ${DATAROOT_VOL}:/data \
		-w /workspace \
		${DOCKER_IMAGE} \
		bash -c "export PYTHONPATH=/workspace && python train.py --config config/$*.yaml"

# Evaluate: make docker-eval-<config-name-without-.yaml>
#   e.g. make docker-eval-eval_unc_distill
docker-eval-%:
	docker run --gpus all --rm \
		-v ${PWD}:/workspace \
		-v ${DATAROOT_VOL}:/data \
		-w /workspace \
		${DOCKER_IMAGE} \
		bash -c "export PYTHONPATH=/workspace && python evaluate_depth.py --config config/$*.yaml"

clean:
	find . -name '*.pyc' -delete; find . -name '__pycache__' -type d -exec rm -rf {} +
