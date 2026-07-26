MNIST Training and Augmentation Experiment

This project keeps the neural-network architecture separate from the trainingworkflows:

backend/ml/models.py defines MNISTClassifier.

backend/ml/train_mnist.py trains the original model without augmentation.

backend/ml/train_mnist_augmented.py trains the same model with controlledaugmentation and reports loss, training accuracy, and test accuracy afterevery epoch.

Install dependencies

From the project root:

python -m pip install torch torchvision

Train the original model

Run:

python backend/ml/train_mnist.py

Results:

Training on: cpu
Epoch 1/5 - Loss: 0.3447
Epoch 2/5 - Loss: 0.1579
Epoch 3/5 - Loss: 0.1080
Epoch 4/5 - Loss: 0.0810
Epoch 5/5 - Loss: 0.0641
Final training accuracy: 98.55%

The original run did not measure test accuracy. The trainer now evaluates theunchanged test dataset as well. Run it again and record the printedFinal test accuracy before completing the fully connected versus CNN table.

Run the augmentation experiment

Run:

python backend/ml/train_mnist_augmented.py

The training pipeline randomly rotates images by up to 10 degrees andtranslates them by up to 10% horizontally and vertically:

train_transform = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomAffine(
        degrees=0,
        translate=(0.1, 0.1),
    ),
    transforms.ToTensor(),
])

The clean training-evaluation set and the test set use onlytransforms.ToTensor().

Results:

Training on: cpu
Training transform: rotation ±10°, translation up to 10%
Epoch 1/5 | Average training loss: 0.7144 | Training accuracy: 93.90% | Test accuracy: 94.05%
Epoch 2/5 | Average training loss: 0.3001 | Training accuracy: 95.58% | Test accuracy: 95.76%
Epoch 3/5 | Average training loss: 0.2253 | Training accuracy: 95.94% | Test accuracy: 96.02%
Epoch 4/5 | Average training loss: 0.1998 | Training accuracy: 96.81% | Test accuracy: 96.72%
Epoch 5/5 | Average training loss: 0.1812 | Training accuracy: 96.75% | Test accuracy: 96.53%

Results comparison

Both experiments use the same MNISTClassifier, batch size of 64, Adamoptimizer, learning rate of 0.001, random seed of 42, and five epochs. Thetraining transformation is the intended experimental difference.

Metric

Original model

Augmented model

Final average training loss

0.0641

0.1812

Final training accuracy

98.55%

96.75%

Final test accuracy

Not measured

96.53%

Best test accuracy

Not measured

96.72% (epoch 4)

Observations

The original model reached a lower training loss and a higher trainingaccuracy than the augmented model.

The augmented model's training loss was higher and its training accuracy was1.80 percentage points lower. This is expected because random rotations andtranslations make the training task more difficult.

The augmented model's test accuracy increased from 94.05% in epoch 1 to96.53% in epoch 5, an improvement of 2.48 percentage points during training.

Its best test accuracy was 96.72% in epoch 4. The small decrease to 96.53% inepoch 5 suggests that performance had started to plateau.

The augmented model's final training and test accuracies differed by only0.22 percentage points, indicating good generalization without substantialoverfitting.

A direct test-accuracy comparison is not yet possible because the originalscript did not evaluate the non-augmented model on the test set. The baselinescript should also report test accuracy before concluding whetheraugmentation improved unseen-data performance.

Augmentation generates a different plausible version of an image whenever itis loaded. It does not permanently add images to the MNIST dataset.

Why test data is not augmented

The test set represents normal unseen data. Randomly changing test images wouldmake evaluation vary from run to run and prevent a stable, fair comparison.Therefore, the test pipeline uses only transforms.ToTensor().

CNN-based MNIST classifier

The CNN experiment is implemented in:

backend/ml/cnn_model.py

backend/ml/train_cnn_mnist.py

The model uses the following architecture:

Conv2d(1, 32) → ReLU → MaxPool2d(2)
Conv2d(32, 64) → ReLU → MaxPool2d(2)
Flatten → Linear → 10 output classes

Training uses CrossEntropyLoss, the Adam optimizer, a batch size of 64, alearning rate of 0.001, and five epochs. The training and test datasets reusethe earlier ToTensor() MNIST data-loading pipeline.

Run the CNN

Run the required kernel_size=3 experiment:

python backend/ml/train_cnn_mnist.py

Run only the kernel_size=5 experiment:

python backend/ml/train_cnn_mnist.py --kernel-size 5

Run both lab experiments and print a measured comparison:

python backend/ml/train_cnn_mnist.py --compare-kernels

The comparison command trains each configuration for five epochs and reports:

final training loss;

final training accuracy;

final test accuracy;

measured training time.

Fully connected versus CNN

Only measured results are recorded. The earlier fully connected run measuredtraining accuracy but did not measure test accuracy, so its 98.55% trainingaccuracy must not be presented as test accuracy. The updated baseline trainernow prints both values.

Model

Measured test accuracy

Notes

Fully connected

Pending baseline rerun

Earlier measured training accuracy was 98.55%

CNN (kernel_size=3)

98.83% final

Best measured accuracy: 98.93% in epoch 4

CNN (kernel_size=5)

99.00% final

Best measured accuracy: 99.14% in epoch 4

Kernel-size lab results

Kernel size

Final training loss

Training accuracy

Test accuracy

Training time

3

0.0313

99.03%

98.83%

129.25 seconds

5

0.0214

99.30%

99.00%

160.74 seconds

Kernel experiment observations

Both configurations were trained for five epochs under the same conditions.

kernel_size=5 achieved a lower final training loss by 0.0099, a highertraining accuracy by 0.27 percentage points, and a higher final test accuracyby 0.17 percentage points.

The best measured test accuracy was 99.14% for kernel_size=5 in epoch 4.The best measured result for kernel_size=3 was 98.93%, also in epoch 4.

Both configurations decreased slightly from their best epoch-4 test accuracyduring epoch 5, suggesting that test performance had started to plateau.

kernel_size=3 completed training in 129.25 seconds. kernel_size=5 took160.74 seconds, making it 31.49 seconds, or approximately 24.4%, slower.

For this experiment, kernel_size=5 performed better when accuracy and finaltraining loss were the priority. kernel_size=3 performed better when trainingspeed was the priority. This conclusion is based only on the measured run above.

The fully connected versus CNN test comparison remains incomplete until theupdated fully connected baseline is rerun and its test accuracy is recorded.

The scripts download MNIST into backend/ml/data/ but do not save modelcheckpoints. Dataset folders, checkpoints (*.pt, *.pth), virtualenvironments, and secrets must remain excluded from Git.