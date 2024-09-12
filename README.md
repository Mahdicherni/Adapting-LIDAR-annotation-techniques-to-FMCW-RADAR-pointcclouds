# Project: Point Cloud Annotation and Network Enhancement for Radar Data Processing

## Overview
This project involves processing radar point cloud data, annotating it, and enhancing a deep learning network to improve radar-specific robustness. The work is centered around the **View-of-Delft (VoD)** dataset and using the  **SalsaNet** model with some modifications.

The primary goal is to detect 3 classes: **background**, **vehicles**, and **road**.
Note that the vod dataset contains 13 different classes: we filter pointcloud data by eliminating many classes. Trucks,motocycles and cars are regrouped in one class named Vehicule.

## 1. Environment Setup
## 0.Environment Setup

It is recommended to use **Anaconda** for managing dependencies and ensuring compatibility across different tools and libraries. To install the appropriate version of TensorFlow for Windows 11, follow these steps:

1. Install the required **CUDA toolkit** and **cuDNN** libraries:
   ```bash
   conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
2.Install TensorFlow version 2.10 (as anything above this is not supported for GPU on Windows Native)
   python -m pip install "tensorflow<2.11"
   # Verify the installation:
   python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

## 1. Data Preparation

We utilized the **View-of-Delft (VoD)** dataset, specifically working with:

- **Bird's Eye View (BEV)** images
- **SFV (Spherical Field of View)** images

### Channels Extracted:
Each image contained the following 4 channels:
- Mean Elevation
- Maximum Elevation
- Average Reflectivity

## 2. Pointwise Annotation

We annotated the point clouds using object bounding boxes derived from camera images. The process involved:

1. **Extracting Extrinsic Parameters**:
   - We extracted the extrinsic parameters of the sensors to map the point clouds to the camera images.
   
2. **Projecting Point Clouds**:
   - Each point in the point cloud was projected onto the camera image. The label of each point was determined by calculating the minimum distance between the point and the center of the bounding box. In cases where a point was within multiple bounding boxes, the closest bounding box determined the label.

## 3. Enhancements to SalsaNet

### TensorFlow Version Upgrade:
- **Upgraded from TensorFlow 1.9 to 2.10** due to incompatibility with Windows 11 and the latest CUDA toolkits.

### Architecture Modifications:
1. **Noise Simulation**:
   - Added a `GaussianNoise` layer to the `resBlock` of the architecture. This simulates noise in radar point clouds, making the network more robust to radar-specific noise.
   
2. **Mixed Precision Training**:
   - Implemented mixed precision training to optimize performance due to limitations in the available GPU vRAM memory.

## Conclusion
By improving the data preparation and making these architectural changes, the network is better equipped to handle radar noise and point cloud data, with optimized performance on modern hardware but further work can be done to ameliorate the sparsity problem.


