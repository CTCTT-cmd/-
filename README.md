# Stringified Image Classifier

Neural network solution for 3-class classification of stringified images.

## Setup

```bash
pip install torch torchvision pandas pillow scikit-learn
```

## Usage

### Train (and generate predictions)

```bash
python train.py \
  --data_dir /path/to/repo \
  --student_id YOUR_STUDENT_ID \
  --backbone efficientnet_b0 \
  --optimizer adamw \
  --epochs 20 \
  --lr 1e-4
```

The script will:
1. Extract the dataset archive automatically on first run.
2. Split training data 80/20 (stratified) into train and validation sets.
3. Fine-tune the chosen backbone on the training split.
4. Save the best weights (`best_model.pth`) based on validation accuracy.
5. Write predictions for the 150 test images to `<student_id>_predictions.csv`.

### Ablation – different optimizers

```bash
python train.py --optimizer sgd   --lr 1e-2
python train.py --optimizer adam  --lr 1e-4
python train.py --optimizer adamw --lr 1e-4
```

### Ablation – different backbones

```bash
python train.py --backbone efficientnet_b0   # ~5.3 M params
python train.py --backbone resnet50          # ~25 M params
python train.py --backbone vit_b_16          # ~86 M params (needs GPU)
```

## Key arguments

| Argument | Default | Description |
|---|---|---|
| `--data_dir` | repo root | Directory with the `.tar.gz` archive |
| `--student_id` | `studentID` | Used as the predictions filename prefix |
| `--backbone` | `efficientnet_b0` | Network architecture |
| `--optimizer` | `adamw` | Optimizer: `adam`, `adamw`, or `sgd` |
| `--lr` | `1e-4` | Learning rate |
| `--epochs` | `20` | Training epochs |
| `--batch_size` | `32` | Mini-batch size |
| `--image_size` | `224` | Input resolution |
| `--val_fraction` | `0.2` | Fraction of train data used for validation |
| `--seed` | `42` | Random seed for reproducibility |

## Output

- `best_model.pth` – saved model weights (best validation accuracy).
- `training_log.csv` – per-epoch loss and accuracy.
- `<student_id>_predictions.csv` – test predictions in the required format (`ID,Label`).
