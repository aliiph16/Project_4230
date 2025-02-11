# Variables
ENV_NAME = ml_env
PYTHON = conda run -n $(ENV_NAME) python
PIP = conda run -n $(ENV_NAME) pip
REQS = requirements.txt
DATA_DIR = data
MODELS_DIR = models
LOGS_DIR = logs
CHECKPOINT_DIR = checkpoints
TENSORBOARD_PORT = 6006
API_PORT = 8080

# Default target: Setup and train
all: setup install train

# === ENVIRONMENT SETUP ===
# Create Conda environment
setup:
	conda create --name $(ENV_NAME) python=3.9 -y

# Install dependencies
install: setup
	conda activate $(ENV_NAME) && $(PIP) install --upgrade pip
	conda activate $(ENV_NAME) && $(PIP) install -r $(REQS)

# === DATA PROCESSING ===
# Download dataset (e.g., from Kaggle)
download:
	$(PYTHON) scripts/download_data.py

# Run full preprocessing pipeline (feature extraction, augmentation, etc.)
preprocess:
	$(PYTHON) scripts/preprocess.py

# === MODEL TRAINING ===
# Train Model 1
train_model1: preprocess
	$(PYTHON) scripts/train.py --model=model1 --save_dir=$(CHECKPOINT_DIR)/model1

# Train Model 2
train_model2: preprocess
	$(PYTHON) scripts/train.py --model=model2 --save_dir=$(CHECKPOINT_DIR)/model2

# Train all models
train: train_model1 train_model2

# === MODEL MANAGEMENT & DEPLOYMENT ===
# Save and load model checkpoints
save_checkpoint:
	$(PYTHON) scripts/save_checkpoint.py --dir=$(CHECKPOINT_DIR)

load_checkpoint:
	$(PYTHON) scripts/load_checkpoint.py --dir=$(CHECKPOINT_DIR)

# Start TensorBoard for logging & monitoring
tensorboard:
	$(PYTHON) -m tensorboard.main --logdir=$(LOGS_DIR) --port=$(TENSORBOARD_PORT)

# Serve model via FastAPI
deploy:
	$(PYTHON) scripts/deploy.py --port=$(API_PORT)

# === TESTING & PERFORMANCE ===
# Run hyperparameter tuning
tune:
	$(PYTHON) scripts/tune_hyperparameters.py

# Profile model performance
profile:
	$(PYTHON) scripts/profile_model.py

# Run all unit & integration tests
test:
	$(PYTHON) -m unittest discover tests

# === MISCELLANEOUS ===
# Git Hook: Run tests before commits
install_git_hooks:
	cp scripts/pre-commit .git/hooks/pre-commit
	chmod +x .git/hooks/pre-commit

# Optimize for parallel execution
parallel_train:
	@echo "Running parallel model training..."
	@make -j 2 train_model1 train_model2

# === CLEANUP ===
# Remove temporary files
clean:
	rm -rf __pycache__ $(DATA_DIR)/*.tmp $(MODELS_DIR)/*.bak $(LOGS_DIR)/* $(CHECKPOINT_DIR)/*

# Remove everything including Conda environment
reset: clean
	conda remove --name $(ENV_NAME) --all -y
	rm -rf $(MODELS_DIR) $(DATA_DIR) $(LOGS_DIR) $(CHECKPOINT_DIR)
