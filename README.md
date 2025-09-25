# [SPiKE: 3D Human Pose from Point Cloud Sequences](https://link.springer.com/chapter/10.1007/978-3-031-78456-9_30)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/spike-3d-human-pose-from-point-cloud/3d-human-pose-estimation-on-itop-front-view-1)](https://paperswithcode.com/sota/3d-human-pose-estimation-on-itop-front-view-1?p=spike-3d-human-pose-from-point-cloud) [![arXiv](https://img.shields.io/badge/arXiv-2409.01879-b31b1b.svg)](https://arxiv.org/abs/2409.01879)

![](https://raw.githubusercontent.com/iballester/spike/main/img/spike.png)

## 📄 Abstract

3D Human Pose Estimation (HPE) is the task of locating key points of the human body in 3D space from 2D or 3D representations such as RGB images, depth maps, or point clouds. Current HPE methods from depth and point clouds predominantly rely on single-frame estimation and do not exploit temporal information from sequences. This paper presents **SPiKE**, a novel approach to 3D HPE using point cloud sequences. Unlike existing methods that process frames of a sequence independently, SPiKE leverages temporal context by adopting a Transformer architecture to encode spatio-temporal relationships between points across the sequence. By partitioning the point cloud into local volumes and using spatial feature extraction via point spatial convolution, SPiKE ensures efficient processing by the Transformer while preserving spatial integrity per timestamp. Experiments on the ITOP benchmark for 3D HPE show that SPiKE reaches **89.19% mAP**, achieving state-of-the-art performance with significantly lower inference times. Extensive ablations further validate the effectiveness of sequence exploitation and our algorithmic choices.

---

## ⚙️ Prerequisites

The code has been tested with the following environment:

- **Python**: 3.8.16
- **g++**: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
- **PyTorch**: 1.8.1+cu111

---

## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/oliver-song-ia/spike
   cd spike
   ```

2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
   ```

3. Compile the CUDA layers required for [PointNet++](http://arxiv.org/abs/1706.02413):
   ```bash
   cd modules
   python setup.py install
   ```

## 📁 Project Structure

Key files for robot planning pipeline:

```
SPiKE/
├── convert_itop_to_training.py     # Config-based data conversion script
├── pose_detector.py                # ROS2 real-time pose detection node
├── generate_predicted_pose.py      # Config-based trajectory generation for robot planning
├── compare_inference_results.py    # Compare inference results between experiments
├── train_itop.py                   # Model training script
├── experiments/Custom/1/
│   └── config.yaml                 # Custom dataset configuration with paths
├── model/
│   ├── model_builder.py            # Model creation with CUSTOM dataset support
│   └── spike.py                    # SPiKE transformer architecture
└── utils/
    └── config_utils.py             # Configuration utilities with path management
```

---

## 📝 How to run SPiKE

### For ITOP Dataset

1. Download the ITOP SIDE dataset (point clouds and labels) from [ITOP Dataset | Zenodo](https://zenodo.org/record/3932973#.Yp8SIxpBxPA) and unzip its contents.

2. Isolate points corresponding to the human body in the point clouds and save the results as `.npz` files.
- You can use the provided script `utils/preprocess_itop.py` as an example. This script takes the original `.h5` files, removes the background by clustering and depth thresholding (see the paper for more details) and saves the results as point cloud sequences in `.npz` format. To run this script, make sure you have the open3d library installed.

3. Update the `ITOP_SIDE_PATH` variable in `const/path` to point to your dataset location. Structure your dataset directory as follows:

   ```
   dataset_directory/
   ├── test/           # Folder containing .npz files for testing
   ├── train/          # Folder containing .npz files for training
   ├── test_labels.h5  # Labels for the test set
   ├── train_labels.h5 # Labels for the training set
   ```

### For Custom Dataset (Robot Planning Pipeline)

1. **Data Conversion**: Convert your cleaned ITOP-format data to training format using the config-based conversion script:
   ```bash
   python convert_itop_to_training.py --config experiments/Custom/1
   ```
   This script reads all paths from the config file and directly converts ITOP format data to training format, including arm labels for robot planner usage. The input data should be located in session subfolders within the configured `dataset_path`.

2. **Model Training**: Train the pose detection model using the configured paths:
   ```bash
   python train_itop.py --config experiments/Custom/1
   ```

3. **Inference and Trajectory Generation**: Use the trained model to predict human poses and generate trajectory CSV files for robot planning:
   ```bash
   python generate_predicted_pose.py --config experiments/Custom/1
   ```
   This generates both predicted pose trajectories (`inference_trajectory.csv`) and ground truth trajectories (`inference_trajectory_gt.csv`) with all 15 joint coordinates plus arm coordinates for robot planner usage.

4. **Real-time ROS2 Integration**: Run the pose detector node for real-time pose detection from point cloud streams:
   ```bash
   python pose_detector.py
   ```
   This subscribes to `/human_pointcloud` topic and publishes skeleton visualization to `/pose_detection` topic in MarkerArray format.

---

## 🚀 Usage

### Training

To train the model, configure your dataset paths in the config file and run:

```bash
python train_itop.py --config experiments/Custom/1
```

### Inference

For predictions on ITOP dataset, update the path pointing to the model weights and run:

```bash
python predict_itop.py --config experiments/ITOP-SIDE/1/config.yaml --model experiments/ITOP-SIDE/1/log/model.pth
```

For trajectory generation and robot planning applications:

```bash
python generate_predicted_pose.py --config experiments/Custom/1
```

You can download our model weights here: [Download Model Weights.](https://cloud.cvl.tuwien.ac.at/s/ATCBp34rH3fGJ23)

### Real-time ROS2 Pose Detection

Launch the pose detector node with custom parameters:

```bash
python pose_detector.py --ros-args -p config_path:=experiments/Custom/1 -p model_path:=experiments/Custom/1/log/best_model.pth -p device:=cuda:0
```

The node will:
- Subscribe to `/human_pointcloud` (PointCloud2 messages)
- Publish skeleton visualization to `/pose_detection` (MarkerArray messages)
- Display real-time inference timing information

### Configuration

The custom dataset configuration (`experiments/Custom/1/config.yaml`) includes:
- Dataset paths: `experiments_path` and `dataset_path`
- Model parameters: transformer depth, heads, dimensions
- Training parameters: batch size, learning rate, epochs
- Joint weights: upper body joints weighted at 1.0, lower body at 0.0
- Data augmentation: center, rotation, and mirror augmentations
