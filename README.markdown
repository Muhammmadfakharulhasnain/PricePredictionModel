# PricePredictionFineTuning

## Overview
This repository contains a machine learning project focused on fine-tuning the Falcon-7B model using LoRA (Low-Rank Adaptation) for price prediction. The project is divided into three Jupyter notebooks:
1. **Notebook 1**: Testing the base Falcon-7B model on a price prediction dataset.
2. **Notebook 2**: Fine-tuning the Falcon-7B model with LoRA on a reduced dataset.
3. **Notebook 3**: Evaluating the fine-tuned model with an improved weighted top-k prediction approach.

The dataset used is `ed-donner/pricer-data` from Hugging Face, containing product descriptions and their prices. The goal is to predict item prices to the nearest dollar based on textual descriptions, with a focus on improving performance through fine-tuning.

## Repository Structure
- `base_model_test.ipynb`: Tests the base Falcon-7B model on the test split of the dataset.
- `model_fine_tuning.ipynb`: Fine-tunes the Falcon-7B model using LoRA with quantization.
- `fine_tuned_model_test.ipynb`: Evaluates the fine-tuned model with a weighted top-k prediction method.
- `README.md`: This file, providing project overview and instructions.
- `requirements.txt`: Lists all necessary Python dependencies.

## Prerequisites
To run the notebooks, you need:
- A Google Colab environment with GPU support (preferably with CUDA 12.4).
- A Hugging Face account with a token (`HF_TOKEN`) for accessing models and datasets.
- A Weights & Biases (`wandb`) account with an API key (`WANDB_API_KEY`) for logging (optional).

### Dependencies
Install the required packages using the following commands:
```bash
pip install -q --upgrade torch==2.5.1+cu124 torchvision==0.20.1+cu124 torchaudio==2.5.1+cu124 --index-url https://download.pytorch.org/whl/cu124
pip install -q --upgrade requests==2.32.3 datasets==3.2.0 peft==0.14.0 trl==0.14.0 matplotlib wandb
pip install --upgrade transformers==4.49.0 accelerate bitsandbytes
```

Alternatively, use the provided `requirements.txt`:
```bash
pip install -r requirements.txt
```

### requirements.txt
```text
torch==2.5.1+cu124
torchvision==0.20.1+cu124
torchaudio==2.5.1+cu124
requests==2.32.3
datasets==3.2.0
peft==0.14.0
trl==0.14.0
matplotlib
wandb
transformers==4.49.0
accelerate
bitsandbytes
```

## Setup
1. **Hugging Face Login**:
   - Set your Hugging Face token in Google Colab secrets as `HF_TOKEN`.
   - Run `login(hf_token, add_to_git_credential=True)` in the notebook.

2. **Weights & Biases Setup** (optional):
   - Set your Weights & Biases API key in Colab secrets as `WANDB_API_KEY`.
   - Initialize `wandb` in the notebook for training logging.

3. **Dataset**:
   - The dataset is loaded from `ed-donner/pricer-data` on Hugging Face.
   - It contains 400,000 training samples and 2,000 test samples with product descriptions and prices.

## Notebooks

### 1. Base Model Testing (`base_model_test.ipynb`)
- **Purpose**: Evaluates the base Falcon-7B model on the test dataset.
- **Key Steps**:
  - Loads the Falcon-7B model with 4-bit quantization.
  - Tests on 250 samples, predicting prices using greedy decoding.
  - Calculates Absolute Error and Squared Logarithmic Error (SLE).
  - Visualizes results with a scatter plot (ground truth vs. predictions).
- **Results**:
  - Mean Absolute Error: $94.10
  - Median Absolute Error: $99.01
  - Mean SLE: 5.984
  - Median SLE: 0.55
  - The base model often predicts round numbers (e.g., $0.00, $29.99), leading to high errors for certain items.

### 2. Model Fine-Tuning (`model_fine_tuning.ipynb`)
- **Purpose**: Fine-tunes the Falcon-7B model using LoRA on a subset of the training data.
- **Key Steps**:
  - Uses a reduced dataset of 500 samples for faster training.
  - Applies LoRA with parameters: `r=32`, `alpha=64`, `dropout=0.1`, targeting Falcon-specific modules (`query_key_value`, `dense`, `dense_h_to_4h`, `dense_4h_to_h`).
  - Training configuration: 1 epoch, batch size 4, learning rate 1e-4, cosine scheduler.
  - Logs training progress to Weights & Biases.
  - Pushes the fine-tuned model to Hugging Face Hub (`fakharbutt44/pricer-2025-08-05_07.12.59`).
- **Notes**:
  - Some dataset entries lack the `Price is $` prefix, causing warnings during training.
  - Consider increasing `max_seq_length` (set to 182) if warnings persist.

### 3. Fine-Tuned Model Testing (`fine_tuned_model_test.ipynb`)
- **Purpose**: Evaluates the fine-tuned model using a weighted top-k prediction method.
- **Key Steps**:
  - Loads the fine-tuned LoRA model from Hugging Face.
  - Uses a top-k (k=3) prediction approach, weighting predictions by token probabilities.
  - Tests on 250 samples, calculating Absolute Error and SLE.
  - Visualizes results with a scatter plot.
- **Results**:
  - Mean Absolute Error: $52.69
  - Median Absolute Error: $54.03
  - Mean SLE: 0.432
  - Median SLE: 0.14
  - The fine-tuned model shows a 44-45% reduction in absolute errors and a 75-93% reduction in SLE compared to the base model.

## Results Summary
The fine-tuned model significantly outperforms the base model:
- **Absolute Error**: Reduced by ~44% (mean) and ~45% (median).
- **SLE**: Reduced by ~93% (mean) and ~75% (median).
- **Observations**:
  - The base model often predicts generic values (e.g., $0.00, $1,000.00), leading to large errors for high/low-value items.
  - The fine-tuned model provides more precise predictions, with stable performance (mean and median errors are close).
  - The weighted top-k prediction method in Notebook 3 improves accuracy by considering multiple probable tokens.

## Visualizations
- Both testing notebooks generate scatter plots comparing ground truth prices to model predictions.
- Colors indicate error magnitude: green (<$40 or <20% error), orange (<$80 or <40% error), red (otherwise).
- The fine-tuned model's scatter plot shows predictions closer to the diagonal (perfect prediction line).

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/fakharbutt44/PricePredictionFineTuning.git
   cd PricePredictionFineTuning
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Open the notebooks in Google Colab or a Jupyter environment with GPU support.
4. Set up `HF_TOKEN` and `WANDB_API_KEY` in Colab secrets.
5. Run the notebooks in order:
   - `base_model_test.ipynb`
   - `model_fine_tuning.ipynb`
   - `fine_tuned_model_test.ipynb`

## Notes
- The fine-tuned model is hosted on Hugging Face at `fakharbutt44/pricer-2025-08-05_07.12.59`.
- The dataset may have inconsistencies (e.g., missing `Price is $` prefix), which can be mitigated by preprocessing or increasing `max_seq_length`.
- Training was performed on a reduced dataset (500 samples) for demonstration; use the full dataset for better performance.
- Ensure CUDA-compatible GPU is available for quantization and training.

## Future Improvements
- Preprocess the dataset to ensure consistent `Price is $` formatting.
- Experiment with higher `max_seq_length` to handle longer inputs.
- Increase training dataset size or epochs for better fine-tuning.
- Try other models (e.g., LLaMA, BLOOM) or adjust LoRA hyperparameters (e.g., `r`, `alpha`).

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact
For questions or contributions, contact [Your Name] at [Your Email] or open an issue on GitHub.