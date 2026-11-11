# Thin wrapper around docker compose. VS Code users can ignore this and just
# run "Dev Containers: Reopen in Container".
COMPOSE := docker compose -f .devcontainer/compose.yaml
SERVICE := mm

.DEFAULT_GOAL := help
.PHONY: help env build up shell check run gz rviz gpu-check down clean

help:  ## Show this help
	@grep -hE '^[a-z0-9-]+:.*?## ' $(MAKEFILE_LIST) | awk -F':.*?## ' '{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

env:   ## Regenerate .devcontainer/.env from this host (UID, DISPLAY)
	@sed -e "s/^USER_UID=.*/USER_UID=$$(id -u)/" \
	     -e "s/^USER_GID=.*/USER_GID=$$(id -g)/" \
	     -e "s/^DISPLAY=.*/DISPLAY=$${DISPLAY:-:0}/" \
	     .devcontainer/.env.example > .devcontainer/.env
	@echo "wrote .devcontainer/.env"; cat .devcontainer/.env

build: ## Build the image
	$(COMPOSE) build

up:    ## Start the container in the background
	@bash scripts/x11-allow.sh
	$(COMPOSE) up -d

shell: up ## Open an interactive shell inside the container
	$(COMPOSE) exec $(SERVICE) bash

check: up ## Preflight: packages, meshes, MoveIt config -- starts nothing
	$(COMPOSE) exec $(SERVICE) bash -lc '$$MOBILE_MANIPULATOR_ROOT/scripts/runRobot.sh --check'

run:   up ## Gazebo tabletop + A200/Kinova/Dex3 + MoveIt + RViz (Ctrl+C stops all)
	$(COMPOSE) exec $(SERVICE) bash -lc '$$MOBILE_MANIPULATOR_ROOT/scripts/runRobot.sh'

gz:    up ## Gazebo alone with the tabletop world
	$(COMPOSE) exec $(SERVICE) bash -lc 'gz sim -r $$MOBILE_MANIPULATOR_ROOT/gazebo_worlds/tabletop.sdf'

rviz:  up ## RViz2 alone with the MoveIt motion-planning view
	$(COMPOSE) exec $(SERVICE) bash -lc 'ros2 launch mobile_manipulator_moveit_config moveit_rviz.launch.py'

gpu-check: up ## Confirm hardware OpenGL is reaching the container
	$(COMPOSE) exec $(SERVICE) bash -lc 'glxinfo -B | grep -E "OpenGL renderer|OpenGL version"'

down:  ## Stop and remove the container
	$(COMPOSE) down

clean: ## Remove the container, image and cache volumes
	$(COMPOSE) down -v --rmi local
