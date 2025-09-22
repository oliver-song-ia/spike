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
   git clone https://github.com/iballester/spike
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

---

## 📝 How to run SPiKE for ITOP

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

---

## 🚀 Usage

### Training

To train the model, check that the config.yaml has the correct parameters and run:

```bash
python train_itop.py --config experiments/ITOP-SIDE/1/config.yaml
```

### Inference

For predictions, update the path pointing to the model weights, check that the config.yaml has the correct parameters and run:

```bash
python predict_itop.py --config experiments/ITOP-SIDE/1/config.yaml --model experiments/ITOP-SIDE/1/log/model.pth
```

You can download our model weights here: [Download Model Weights.](https://cloud.cvl.tuwien.ac.at/s/ATCBp34rH3fGJ23)

---

## 🎥 Qualitative Results

For video samples showcasing pose predictions on the testing set, please visit:

- [Video 1](https://youtu.be/mk_UffjtTlM)
- [Video 2](https://youtu.be/YZXXY0DLQWo)
- [Video 3](https://youtu.be/8j7yt-1sToU)
- [Video 4](https://youtu.be/ZQQSviiT7Sw)
- [Video 5](https://youtu.be/MvvgQYlsYlY)
- [Video 6](https://youtu.be/IMvdci9RgAM)

---

## 📚 Citation

If you find our work useful, we'd really appreciate it if you cite us:

```bibtex
@inproceedings{ballester2024spike,
  title={SPiKE: 3D Human Pose from Point Cloud Sequences},
  author={Ballester, Irene and Peterka, Ond{\v{r}}ej and Kampel, Martin},
  booktitle={Pattern Recognition},
  year={2024}
}
```

---

## 🙏 Acknowledgments

A big thanks to the following open-source projects for their contributions:

1. [PointNet++ PyTorch Implementation (VoteNet)](https://github.com/facebookresearch/votenet/tree/master/pointnet2)
2. [P4Transformer](https://github.com/hehefan/P4Transformer)

Their work greatly facilitated the development of this project.

---

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.
