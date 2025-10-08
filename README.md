

# CIFAR-10 Image Classifier with PyTorch

A simple Convolutional Neural Network (CNN) built using PyTorch to classify images from the CIFAR-10 dataset. This project achieves a test accuracy of **78.1%** and includes training visualizations and a confusion matrix.

##  Features
- CNN with 3 convolutional layers + dropout
- Checkpoint saving/loading (`checkpoint.pth`)
- Training loss and accuracy tracking
- Final test accuracy report
- Confusion matrix visualization

##  Dataset
[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) — 60,000 32x32 color images in 10 classes:

plane, car, bird, cat, deer, dog, frog, horse, ship, truck

##  Results
- ✅ **Test Accuracy:** 78.1%
-  Training loss and accuracy plotted
-  Confusion matrix generated

##  How to Run
1. Clone the repo:
   ```bash
   git clone https://github.com/SyntaxNomad/CNN-Image-Recognition-pytorch
   cd CNN-Image-Recognition-pytorch

	2.	Install dependencies: pip install torch torchvision matplotlib scikit-learn


	3.	Run training: python main.py



 Notes
	•	Training resumes from saved checkpoint if it exists.
	•	Uses data augmentation via RandomHorizontalFlip.
