# ‎‎
#  __   __               _        _      _      __  __ 
#  \ \ / /    /\        | |      | |    | |    |  \/  |
#   \ V /    /  \  _   _| |_ ___ | |    | |    | \  / |
#    > <    / /\ \| | | | __/ _ \| |    | |    | |\/| |
#   / . \  / ____ \ |_| | || (_) | |____| |____| |  | |
#  /_/ \_\/_/    \_\__,_|\__\___/|______|______|_|  |_|
#
# Usage: make [command]

# help         Show this information.
.PHONY: help
help:
	cat makefile | grep -oP "^# \K(.*)"

# clean        Remove (!) all untracked and ignored files.
.PHONY: clean
clean:
	git clean -xdff

# ‎‎
# ---------------------------------------------------------------------------
# The following commands must be run OUTSIDE the development environment.
# ---------------------------------------------------------------------------
# ‎‎

# docker-base  Builds the development base image from scratch.
.PHONY: docker
docker:
	docker build . -t autogoal/xautollm:latest -f dockerfiles/core/dockerfile --no-cache

# container          Creates the base development container.
.PHONY: container
container:
	docker-compose run --service-ports --name=xautollm-gpu xautollm-gpu

# container         Run the base development image with GPU enabled.
.PHONY: container-gpu
container-gpu:
	docker run -it --name xautollm-gpu --hostname xautollm-gpu -p 8500:8501 -p 8000:8000 -v "$(pwd):/home/coder/autogoal" -v "$HOME/.ssh:/home/coder/.ssh" -v "$HOME/.gitconfig:/home/coder/.gitconfig" --user 1000:1000 --gpus all autogoal/xautollm:latest

# ‎‎
# ---------------------------------------------------------------------------
# The following commands must be run INSIDE the development environment.
# ---------------------------------------------------------------------------
# ‎‎

.PHONY: ensure-dev
ensure-dev:
	echo ${BUILD_ENVIRONMENT} | grep "development" >> /dev/null

.PHONY: docs-dev
docs-dev:
	python3 -m illiterate autogoal/autogoal autogoal/docs/api
	python3 -m illiterate autogoal/tests/examples autogoal/docs/examples
	python3 -m illiterate autogoal/tests/guide autogoal/docs/guide
	cp Readme.md autogoal/docs/index.md
	# python3 -m typer_cli autogoal/autogoal/__main__.py utils autogoal/docs > autogoal/docs/cli-api.md

# docs         Compile and publish the documentation to Github.
.PHONY: docs
docs: ensure-dev docs-dev
	mkdocs build

# gh-deploy    Deploy docs to Github Pages
.PHONY: gh-deploy
gh-deploy: ensure-dev docs-dev
	git remote add pages git@github.com:autogoal/autogoal.github.io || echo "remote exists"
	mkdocs gh-deploy -r pages -b master --force

# format       Format all source code inplace using `black`.
.PHONY: format
format: ensure-dev
	(git status | grep "nothing to commit") && sudo black autogoal/ tests/ || echo "(!) REFUSING TO REFORMAT WITH UNCOMMITED CHANGES" && exit
	git status

# anim         Make CLI animations
.PHONY: anim
anim:
	termtosvg -c "bash docs/shell/autogoal_cli.sh" docs/shell/autogoal_cli.svg -g 80x20 -m 100 -t window_frame_powershell

# env          Setup the development environment.
.PHONY: env
env: ensure-dev
	curl -sSL https://install.python-poetry.org | python3 -
	ln -s ${HOME}/.local/share/pypoetry /usr/bin/poetry
	poetry config virtualenvs.create false

# install      Install all the development dependencies.
.PHONY: install
install: ensure-dev
	poetry install


# ‎‎
# ---------------------------------------------------------------------------
# The following commands are for remote communication between enviroments.
# ---------------------------------------------------------------------------
# ‎‎


.PHONY: host-ns
host-ns: ensure-dev
	pyro5-ns 

# ‎‎
# ---------------------------------------------------------------------------
# The following commands are for testing in the development enviroment.
# ---------------------------------------------------------------------------
# ‎‎

# test-core    Run the core unit tests (not contrib).
.PHONY: test-core
test-core: ensure-dev
	pytest autogoal/tests/core

# test-contrib Run the contrib unit tests.
.PHONY: test-contrib
test-contrib: ensure-dev
	bash scripts/run_all_contrib_tests.sh

# test-specific-contrib Run any specific contrib unit tests.
test-specific-contrib:
	bash scripts/run_specific_contrib_tests.sh $(CONTRIB)
	
# test-sklearn Run the sklearn contrib unit tests.
.PHONY: test-sklearn
test-sklearn: 
	make test-contrib CONTRIB=sklearn

# test-nltk    Run the nltk contrib unit tests.
.PHONY: test-nltk
test-nltk: 
	make test-contrib CONTRIB=nltk

# test-full    Run all unit tests including the (very) slow ones.
.PHONY: test-full
test-full: ensure-dev test-contrib
	python -m pytest autogoal tests/core tests/contrib --ignore=autogoal/datasets --ignore=autogoal/experimental -v

# cov          Run the coverage analysis.
.PHONY: cov
cov: ensure-dev
	python -m codecov

# ‎‎
