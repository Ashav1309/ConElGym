from src.models.hyperparameter_tuning import tune_hyperparameters

if __name__ == "__main__":
    best_params = tune_hyperparameters()
    print("Best parameters:", best_params) 