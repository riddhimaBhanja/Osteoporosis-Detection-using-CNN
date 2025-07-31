# Osteoporosis Classification from OPG Images

## Dataset

The dataset used in this project consists of orthopantomogram (OPG) images acquired by a dentist. **Due to privacy and ethical considerations, the dataset is not publicly shared, as the dentist has not approved its distribution.**

## Preprocessing

The OPG images undergo a sequential preprocessing pipeline to enhance features for classification:
1. **CLAHE (Contrast Limited Adaptive Histogram Equalization)**: Applied to improve contrast in the images, enhancing visibility of bone structures.
2. **Canny Edge Detection**: Utilizes a multi-stage algorithm with Gaussian smoothing, gradient computation, non-maximum suppression, and double thresholding (set at 100 and 200) to detect strong edges.
3. **Sobel Edge Detection**: Computes gradient magnitudes in the horizontal and vertical directions using a Sobel operator (kernel size=1), combining absolute gradients into a single uint8 image to highlight edge intensity.

## Methodology

### Technology Used

The methodology processes an **unlabeled dataset** of OPG images for osteoporosis classification using a convolutional neural network (CNN). The process is as follows:

**Data Preparation and Augmentation**: The unlabeled dataset is organized into two age groups: 17-40 and above 40, stored in a compressed archive and extracted to a designated directory. Images are converted to grayscale and resized to 128x128 using a computer vision library. A custom dataset class, built on a deep learning framework, applies transformations including normalization (mean 0.5, standard deviation 0.5), conversion to floating-point tensors, and augmentation techniques (random rotation by 10 degrees, horizontal flipping, and affine scaling between 0.9 and 1.1) to enhance training robustness.

**Pseudo-Labeling and Data Splitting**: Pseudo-labels are assigned to the unlabeled data, with images from the 17-40 age group labeled as normal (0) and those from the above 40 group labeled as osteoporotic (1) with an 85% probability, reflecting the assumed prevalence of osteoporosis. A stratified 80/20 train-validation split is performed using a machine learning utility to maintain pseudo-label distribution.

**CNN Architecture**: The CNN, implemented in a deep learning framework, consists of three convolutional layers (1→16, 16→32, 32→64 channels, 3x3 kernels, padding=1), each followed by ReLU activations and max-pooling (2x2, stride=2), reducing spatial dimensions (128x128→64x64→32x32→16x16). The feature maps are flattened and processed through fully connected layers (64*16*16→128, 128→2) with ReLU activation and a dropout rate of 0.5. The model outputs probabilities for normal and osteoporotic classes via an implicit softmax.

**Training and Optimization**: Training is conducted on a CUDA-enabled GPU, if available, using device-agnostic computation. The Adam optimizer (learning rate=0.001) and a weighted cross-entropy loss (weights [0.15, 0.85]) address pseudo-label imbalance. Mixed precision training enhances efficiency, and gradient clipping (max norm=1.0) prevents exploding gradients. The model is trained for 50 epochs, with validation accuracy monitored to save the best checkpoint.

**Testing and Classification**: Thirteen test images are processed without augmentation. The model predicts osteoporosis probabilities, adjusted by image intensity (mean pixel value scaled by 0.15), with likelihoods constrained between 0 and 1. Images with likelihoods above 0.5 are classified as osteoporotic and saved to an `osteoporosis` directory, while others are saved to a `normal` directory. A summary text file reports the counts of normal and osteoporotic images and their respective likelihood scores.

**Data Management**: The dataset is accessed from a cloud storage service, copied to the local environment, and classification results are transferred to a specified cloud directory for persistent storage, with robust directory creation to handle existing or missing paths.

### Tools and Libraries
- **Python**: Core programming language.
- **OpenCV (cv2)**: For image preprocessing (grayscale conversion, resizing, CLAHE, Canny, and Sobel edge detection).
- **PyTorch**: For CNN model development, training, and inference.
- **scikit-learn**: For stratified train-validation splitting.
- **NumPy**: For numerical operations and intensity calculations.
- **pathlib**: For cross-platform file path handling.
- **shutil, os, zipfile**: For file and directory management, including dataset extraction and result storage.

### Requirements

To run the pipeline, install the following dependencies in a Python environment (e.g., Google Colab):
```bash
pip install torch torchvision opencv-python-headless numpy scikit-learn
