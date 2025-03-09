# Makefile for XAutoLLM experiments (AutoGOAL-based)

# Define environment variables for experience store location
export AUTOGOAL_EXPERIENCE_DIR=/home/coder/.autogoal/data/experience_store
export ANALYSIS_EXPERIENCE_DIR=/home/coder/autogoal/experiments/data/experience_store

# Default target runs both multi-objective and single-objective experiments in the correct order
all: run-multi-objective run-single-objective analyze-results

# Run multi-objective experiments first (generates experience in ~/.autogoal/data/experience_store)
run-multi-objective:
	@echo "Running multi-objective experiments (experience will be stored in $(AUTOGOAL_EXPERIENCE_DIR))"
	python src/execute_experiments.py --experiment_type multi

# Run single-objective experiments after multi-objective (generates experience in ~/.autogoal/data/experience_store)
run-single-objective:
	@echo "Running single-objective experiments (experience will be stored in $(AUTOGOAL_EXPERIENCE_DIR))"
	python src/execute_experiments.py --experiment_type single

# Run both multi and single-objective experiments in the correct order
run-experiments: run-multi-objective run-single-objective

# Analyze multi-objective results using the analysis module
analyze-multi-objective:
	@echo "Analyzing multi-objective experiment results (using experience from $(ANALYSIS_EXPERIENCE_DIR))"
	python src.analysis.multi_objective.main.py

# Analyze single-objective results using the analysis module
analyze-single-objective:
	@echo "Analyzing single-objective experiment results (using experience from $(ANALYSIS_EXPERIENCE_DIR))"
	python src.analysis.single_objective.main.py

# Analyze both multi and single-objective results
analyze-results: analyze-multi-objective analyze-single-objective

# Generate paper results (runs everything in the correct order)
paper: all

# Copy experiences from default location to analysis location
copy-experiences:
	@echo "Copying experiences from $(AUTOGOAL_EXPERIENCE_DIR) to $(ANALYSIS_EXPERIENCE_DIR)"
	mkdir -p $(ANALYSIS_EXPERIENCE_DIR)
	cp -r $(AUTOGOAL_EXPERIENCE_DIR)/* $(ANALYSIS_EXPERIENCE_DIR)/

# Clean output files
clean:
	rm -rf output/logs/*.log
	rm -rf output/*.json
