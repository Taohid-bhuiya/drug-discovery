import os
import subprocess
import logging
import pandas as pd
from sklearn.metrics import precision_recall_curve, auc, f1_score

# Set global paths
data_dir = './data'  # Path to the folder containing test CSV
output_dir = './output'  # Path to save model outputs

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Logger setup
def setup_logger(output_dir):
    log_file = os.path.join(output_dir, 'pipeline.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("ChempropLogger")

logger = setup_logger(output_dir)

def train_with_scaffold_split_and_ensemble(test_path, output_dir):
    """Train the model with scaffold splitting, ensemble, and replicates."""
    model_output_dir = os.path.join(output_dir, 'trained_model')
    os.makedirs(model_output_dir, exist_ok=True)

    # Train the model with scaffold splitting, ensemble, and multiple replicates
    train_command = [
        "chemprop",
        "train",
        "--data-path", test_path,
        "--task-type", "classification",
        "--split", "scaffold_balanced",
        "--ensemble-size", "5",  # Use ensemble with 5 models
        "--num-replicates", "5",  # Perform 5 replicates
        "--save-dir", model_output_dir
    ]

    logger.info(f"Running command: {' '.join(train_command)}")
    subprocess.run(train_command, check=True)

    return model_output_dir

def predict_and_calculate_metrics(test_path, model_output_dir, output_dir):
    """Run predictions and calculate metrics."""
    predictions_path = os.path.join(output_dir, 'predictions.csv')

    # Locate the best model checkpoint(s) from the directory
    checkpoint_paths = []
    for root, _, files in os.walk(model_output_dir):
        for file in files:
            if file == "best.pt":
                checkpoint_paths.append(os.path.join(root, file))

    if not checkpoint_paths:
        logger.error("No model checkpoints (best.pt) found in the specified directory.")
        raise FileNotFoundError("Ensure training was successful and checkpoints exist.")

    # Predict using the trained model
    predict_command = [
        "chemprop",
        "predict",
        "--test-path", test_path,
        "--model-paths", *checkpoint_paths,  # Pass all best.pt paths
        "--preds-path", predictions_path
    ]

    logger.info(f"Running command: {' '.join(predict_command)}")
    subprocess.run(predict_command, check=True)

    # Load predictions and test data
    predictions = pd.read_csv(predictions_path)
    test_data = pd.read_csv(test_path)

    # Debug: Print column names
    print("Predictions file columns:", list(predictions.columns))
    print("Test file columns:", list(test_data.columns))

    # Strip any potential leading/trailing spaces from column names
    predictions.columns = predictions.columns.str.strip()
    test_data.columns = test_data.columns.str.strip()

    # Ensure columns match between the files
    if 'Active' not in predictions.columns or 'SMILES' not in predictions.columns:
        logger.error("Predictions file does not contain required columns: 'Active' or 'SMILES'.")
        raise KeyError("Ensure 'Active' and 'SMILES' columns exist in predictions.csv.")

    # Merge predictions with actual values
    merged_data = predictions.merge(test_data, on="SMILES")

    # Extract ground truth and predicted probabilities
    ground_truth = merged_data["Active_y"]  # 'Active_y' comes from the test file during merge
    predicted_probs = merged_data["Active_x"]  # 'Active_x' comes from predictions file

    # Calculate Precision-Recall Curve
    precision, recall, thresholds = precision_recall_curve(ground_truth, predicted_probs)

    # Calculate F1 scores for each threshold
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)  # Avoid division by zero
    best_threshold_idx = f1_scores.argmax()
    optimum_threshold = thresholds[best_threshold_idx] if len(thresholds) > 0 else 0.5

    # Calculate F1 Score for the optimum threshold
    predicted_labels = (predicted_probs > optimum_threshold).astype(int)
    optimum_f1_score = f1_score(ground_truth, predicted_labels)

    # Log detailed F1 threshold exploration
    for idx, threshold in enumerate(thresholds):
        logger.info(f"Threshold: {threshold:.4f}, Precision: {precision[idx]:.4f}, Recall: {recall[idx]:.4f}, F1: {f1_scores[idx]:.4f}")

    # Log optimum threshold and corresponding F1 score
    logger.info(f"Optimum Threshold: {optimum_threshold}")
    logger.info(f"F1 Score at Optimum Threshold: {optimum_f1_score}")

    return optimum_threshold, optimum_f1_score



if __name__ == "__main__":
    try:
        # Step 1: Prepare test dataset
        test_path = os.path.join(data_dir, 'test.csv')
        if not os.path.exists(test_path):
            logger.error("Test file not found in the data directory.")
            raise FileNotFoundError("Ensure test.csv is in the data directory.")

        # Step 2: Train model with scaffold split, ensemble, and replicates
        logger.info("Training with scaffold splitting, ensemble, and replicates...")
        model_output_dir = train_with_scaffold_split_and_ensemble(test_path, output_dir)

        # Step 3: Predict and calculate metrics
        logger.info("Predicting and calculating metrics...")
        auprc, f1 = predict_and_calculate_metrics(test_path, model_output_dir, output_dir)

        logger.info(f"Metrics: AUPRC={auprc}, F1={f1}")

    except Exception as e:
        logger.error(f"An error occurred: {e}")
