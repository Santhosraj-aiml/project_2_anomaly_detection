**Visual Anomaly Detection on MVNet Dataset**

**demo video**



**Objective:**

Develop an anomaly detection system using computer vision techniques to identify and localize visual defects in images from the MVNet dataset. The system should classify images and, if possible, highlight the anomaly region through segmentation or masking.

**2. Dataset**: MVNet Dataset

The dataset consists of multiple categories of industrial objects, including bottle, leather, metal, transistor, etc. The project can either consider all categories or focus on a specific category for analysis.

**Data Split:**

Training Set: 80% of the dataset

Testing Set: 20% of the dataset

Batch Size: 32


**3. Feature Extraction**

Backbone Model: ResNet-50 (pretrained on ImageNet) is used to extract high-level feature representations from images.

Feature Layers Used: Intermediate layers (Layer 2 and Layer 3) are selected for feature extraction.

Processing: Extracted features are resized using adaptive average pooling before feeding them into the anomaly detection model.

**4. Anomaly Detection Model**

A deep learning-based Autoencoder is employed for anomaly detection:

Encoder: Reduces dimensionality by compressing feature maps.

Decoder: Reconstructs the original feature maps from the latent representation.

Loss Function: Mean Squared Error (MSE) to quantify reconstruction loss.

Optimizer: Adam with a learning rate of 0.001.

**5. Model Training**

The Autoencoder is trained for 100 epochs using extracted features.

Training Strategy:

The model learns to reconstruct features from normal samples.

Training and validation losses are monitored.

**6. Anomaly Detection Process**

Extract features from the test image using ResNet-50.

Pass features through the trained autoencoder for reconstruction.

Compute reconstruction error using Mean Squared Error (MSE).

Apply a decision function to classify an image as normal or anomalous.

Generate an anomaly segmentation map (if applicable) to visualize the defect.

**7. Thresholding for Decision Making**

Compute the mean and standard deviation of reconstruction errors on normal samples.

Define a threshold: If the reconstruction error exceeds this threshold, the image is classified as anomalous.

**8. Model Evaluation**

**The system is evaluated using:**

Precision, Recall, and F1-score: Evaluates the effectiveness of anomaly detection.

Heatmaps for Visualization: Highlight regions where anomalies are detected.

**9. Future Enhancements**

Integration of Additional Computer Vision Techniques: Test other architectures such as Vision Transformers (ViTs) or GAN-based anomaly detection.

Improve Localization: Implement advanced segmentation methods like U-Net to precisely localize defects.

Real-time Processing: Optimize the model for deployment in industrial applications.

