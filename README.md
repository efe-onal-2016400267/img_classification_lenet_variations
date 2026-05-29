# Image Classification with LeNet-5 Variations

---

## 📌 Overview

This project is a passion-driven exploration into **Deep Learning, Convolutional Neural Networks (CNNs), Transfer Learning, and Model Interpretability**. Using the classic **LeNet-5** architecture as a foundation, this project designs, trains, and evaluates multiple architectural variations on the **CIFAR-10** and **CIFAR-100** datasets. 

Beyond standard model training, this project incorporates advanced regularization techniques (Batch Normalization, Dropout), learning rate scheduling, cross-dataset transfer learning, rigorous statistical significance testing (t-tests), and deep interpretability visualizations (t-SNE and filter activation maps).

---

## 🚀 Key Features

* **Architectural Evolution**: Implements and compares three distinct variations of LeNet-5:
  1. **Baseline LeNet-5**: The classic architecture with Average Pooling and Kaiming Uniform initialization.
  2. **LeNet with Batch Normalization**: Replaces average pooling with max pooling and adds 2D/1D Batch Normalization layers to accelerate convergence.
  3. **LeNet with Batch Normalization & Dropout**: Integrates Dropout layers ($p=0.2$) to mitigate overfitting and improve generalization.
* **Data Augmentation Pipeline**: Expands the training dataset by combining original images with horizontally flipped, vertically flipped, and rotated ($30^\circ$) variations.
* **Transfer Learning & Fine-Tuning**: Trains the best-performing architecture on CIFAR-100 (mapped to 20 coarse superclasses) and fine-tunes it on CIFAR-10 to evaluate cross-dataset feature transferability.
* **Rigorous Statistical Evaluation**: Runs each model variation 10 times with randomized data splits and performs independent **paired t-tests** to verify if performance gains are statistically significant.
* **Model Interpretability & Visualizations**:
  * **Filter Activation Maps**: Extracts and visualizes the top 10 images and corresponding feature maps that maximize activations for each filter in the convolutional layers.
  * **t-SNE Feature Space Projection**: Projects high-dimensional features from the final fully connected layer into 2D space to visualize class clustering.
  * **Confusion Matrices**: Generates detailed confusion matrices to analyze class-specific classification performance.

---

## 📂 Repository Structure

```bash
├── src/
│   └── img_classification_lenet.ipynb  # Main Jupyter Notebook containing all experiments
├── requirements.txt                    # Project dependencies
└── README.md                           # Project documentation
```

---

## 🛠️ Installation & Setup

### Prerequisites
Ensure you have Python 3.8+ installed. Install the required dependencies using the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```

#### `requirements.txt` Contents:
```text
torch>=1.10.0
torchvision>=0.11.0
torcheval>=0.0.1
torchmetrics>=0.8.0
torchsummary>=1.5.1
numpy>=1.21.0
scikit-learn>=0.24.0
scipy>=1.7.0
matplotlib>=3.4.0
seaborn>=0.11.0
```

---

## 🚀 How to Run

All experiments, training loops, visualizations, and statistical analyses are self-contained within the Jupyter Notebook:

1. Open the notebook:
   ```bash
   jupyter notebook src/img_classification_lenet.ipynb
   ```
2. Run the cells sequentially to:
   * Load and preprocess CIFAR-10/CIFAR-100.
   * Define and train the LeNet-5 variations.
   * Generate training/validation loss and accuracy curves.
   * Perform transfer learning and fine-tuning.
   * Visualize filter activations, t-SNE projections, and confusion matrices.
   * Execute the statistical t-tests.

---

## 📐 Technical Details & Methodology

### A. Model Architectures

#### 1. Baseline LeNet-5 (`Lenet`)
* **Conv Layers**: Three convolutional layers with $5\times5$ kernels.
* **Activation**: ReLU.
* **Pooling**: Average Pooling ($2\times2$, stride 2).
* **FC Layers**: Two fully connected layers ($120 \to 84 \to 10$).

#### 2. LeNet with Batch Normalization (`Lenet_bn`)
* Adds `BatchNorm2d` after convolutional layers and `BatchNorm1d` after the first fully connected layer.
* Replaces Average Pooling with Max Pooling to capture more prominent features.

#### 3. LeNet with Batch Normalization & Dropout (`Lenet_dropout`)
* Adds `Dropout` layers ($p=0.2$) after the second convolutional layer and the first fully connected layer to prevent co-adaptation of features.

### B. Transfer Learning Pipeline
1. **Pre-training**: Train the `Lenet_dropout` model on CIFAR-100, mapping the 100 fine classes to 20 coarse superclasses using a custom mapping dictionary.
2. **Fine-tuning**: Replace the final classification head (`fc2`) to output 10 classes instead of 20. Fine-tune the entire network on CIFAR-10 using a halved learning rate ($0.0005$) and early stopping.

### C. Interpretability Visualizations
* **t-SNE**: Extracts the 84-dimensional feature vector from the first fully connected layer (`fc1`) for all test images, projects them to 2D using t-SNE, and colors them by class label.
* **Filter Activations**: Iterates through the test dataset to find the top 10 images that produce the highest summed activation for each filter in `Conv1` and `Conv2`. Plots the original image alongside the corresponding feature map.

---

## 📊 Results & Insights

* **Regularization Impact**: The addition of Batch Normalization and Dropout significantly accelerates convergence and reduces overfitting, leading to higher validation accuracy compared to the baseline LeNet-5.
* **Transfer Learning**: Pre-training on CIFAR-100 coarse classes provides a robust feature extractor, allowing the fine-tuned model to achieve competitive accuracy on CIFAR-10 with fewer training epochs.
* **Statistical Significance**: The paired t-tests confirm that the performance improvements of the Batch Normalization and Dropout variants over the baseline model are statistically significant ($p < 0.05$).

---

## 👥 Authors & Contributors

* **Efe Önal**
* **Arda Canser Adalı**

