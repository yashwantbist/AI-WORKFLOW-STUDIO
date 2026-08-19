ResNet18 Transfer Learning

This lab reuses a ResNet18 model pretrained on ImageNet to classify CIFAR-10images. The pretrained backbone already recognizes useful visual features suchas edges, textures, and shapes. Only the layers needed for the target task aretrained.

Files

transfer_learning.py loads ResNet18, freezes its backbone, replaces thefinal classifier, verifies frozen layers, and optionally unfreezes layer4.

train_transfer_learning.py downloads CIFAR-10, trains for five epochs,evaluates the test subset after every epoch, measures training time, andprints the experiment comparison.

../tests/test_transfer_learning.py verifies both layer-freezingconfigurations without downloading pretrained weights.

Architecture

CIFAR-10 image
    ↓
Resize and ImageNet normalization
    ↓
ResNet18 pretrained feature extractor
    ↓
New Linear classifier (10 classes)
    ↓
CIFAR-10 prediction

The real training command always usesResNet18_Weights.DEFAULT. PyTorch downloads the pretrained weights to itslocal cache; this project does not save or commit them.

Install

From the project root:

python -m pip install -r backend/ml/requirements-ml.txt

Experiment 1: frozen backbone

This is the default run. Every backbone parameter is frozen and only the newfc classifier is trained:

python backend/ml/train_transfer_learning.py

Experiment 2: fine-tune the final residual block

This unfreezes layer4 as well as the fc classifier:

python backend/ml/train_transfer_learning.py --strategy fine-tune-layer4

Run the complete comparison lab

This command runs both strategies for five epochs and prints a comparison ofloss, training accuracy, test accuracy, trainable parameter count, and measuredtraining time:

python backend/ml/train_transfer_learning.py --compare-strategies

The default lab uses a deterministic random subset of 10,000 training imagesand 2,000 test images, resized to 128 × 128. This keeps both experimentspractical on a CPU while ensuring they use the same examples and shuffle seed.For a faster pipeline check before the full lab, run:

python backend/ml/train_transfer_learning.py `
  --compare-strategies `
  --epochs 1 `
  --train-samples 500 `
  --test-samples 100

Run on all 50,000 training and 10,000 test images with:

python backend/ml/train_transfer_learning.py `
  --compare-strategies `
  --train-samples 0 `
  --test-samples 0

On Windows, keep --num-workers 0 unless multiprocessing is configured. Ifmemory is limited, reduce --batch-size from 64 to 32 or 16.

Results

Record only values printed by your completed run:

Strategy

Final loss

Training accuracy

Test accuracy

Training time

Classifier head only

Pending measured run

Pending measured run

Pending measured run

Pending measured run

Fine-tune layer4 + head

Pending measured run

Pending measured run

Pending measured run

Pending measured run

Do not replace the pending cells with estimated results. After the labfinishes, copy the final comparison table printed by the script into thissection.

What is excluded from Git

The repository .gitignore excludes:

backend/ml/data/, including the downloaded CIFAR-10 files;

model checkpoints ending in .pt or .pth;

virtual environments and .env secrets.

The trainer reports metrics to the terminal and does not save model weights.