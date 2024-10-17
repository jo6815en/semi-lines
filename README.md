## Getting Started

### Installation

```bash
cd UniMatch
conda create -n unimatch python=3.10.4
conda activate unimatch
pip install -r requirements.txt
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html
```

### Pretrained Backbone

[ResNet-50](https://drive.google.com/file/d/1mqUrqFvTQ0k5QEotk4oiOFyP6B9dVZXS/view?usp=sharing) | [ResNet-101](https://drive.google.com/file/d/1Rx0legsMolCWENpfvE2jUScT3ogalMO8/view?usp=sharing) | [Xception-65](https://drive.google.com/open?id=1_j_mE07tiV24xXOJw4XDze0-a0NAhNVi)

```
├── ./pretrained
    ├── resnet50.pth
    ├── resnet101.pth
    └── xception.pth
```

### Dataset

- Pascal: [JPEGImages](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar) | [SegmentationClass](https://drive.google.com/file/d/1ikrDlsai5QSf2GiSUR3f8PZUzyTubcuF/view?usp=sharing)
- Cityscapes: [leftImg8bit](https://www.cityscapes-dataset.com/file-handling/?packageID=3) | [gtFine](https://drive.google.com/file/d/1E_27g9tuHm6baBqcA7jct_jqcGA89QPm/view?usp=sharing)
- COCO: [train2017](http://images.cocodataset.org/zips/train2017.zip) | [val2017](http://images.cocodataset.org/zips/val2017.zip) | [masks](https://drive.google.com/file/d/166xLerzEEIbU7Mt1UGut-3-VN41FMUb1/view?usp=sharing)

Please modify your dataset path in configuration files.

**The groundtruth masks have already been pre-processed by us. You can use them directly.**

```
├── [Your Pascal Path]
    ├── JPEGImages
    └── SegmentationClass
    
├── [Your Cityscapes Path]
    ├── leftImg8bit
    └── gtFine
    
├── [Your COCO Path]
    ├── train2017
    ├── val2017
    └── masks
```

## Usage

### Docker

## Pull the Docker Image

Pull the Docker image from Docker Hub using the following command:

```bash
docker pull johannaengman/semi-lines:latest
```

This will fetch the latest version of the Docker image.

## Running the Container with Mounted Data

To run the container and mount your local data, use the following command:

```bash
docker run -it --gpus all -v /path/to/your/data:/home2/johannae/semi-lines/UniMatch johannaengman/semi-lines:latest
```

### Explanation:
- ` /path/to/your/data`: Replace with the directory of your code and data. 
- `-it`: Runs the container interactively.
- `-v /path/to/your/data:/app/data`: Mounts your local directory `/path/to/your/data` to the `/app/data` directory inside the container. Replace `/path/to/your/data` with the actual path to your dataset.

### Running the Bash Script

Once the container is running and your data is mounted, execute the bash script inside the container:

```bash
bash run_script.sh
```

This script will process the data and run the necessary code based on the dataset you’ve mounted.

## Example

Here’s an example of a full command:

```bash
docker run -it --rm -v ~/mydata:/app/data johannaengman/semi-lines:latest bash run_script.sh
```

In this example, the `mydata` directory from your home folder is mounted into the container's `/app/data` folder, and the `run_script.sh` script is executed.

## Stopping the Container

The container will stop automatically when the script finishes. If you need to stop it manually, use:

```bash
docker stop <container-id>
```

## Building the Image Locally (Optional)

If users prefer to build the image themselves, provide the build instructions:

```bash
git clone https://github.com/your-repo.git
cd your-repo
docker build -t johannaengman/semi-lines .
```



### UniMatch

```bash
# use torch.distributed.launch
sh scripts/train.sh <num_gpu> <port>

# or use slurm
# sh scripts/slurm_train.sh <num_gpu> <port> <partition>
```

To train on other datasets or splits, please modify
``dataset`` and ``split`` in [train.sh](https://github.com/LiheYoung/UniMatch/blob/main/scripts/train.sh).

### FixMatch

Modify the ``method`` from ``'unimatch'`` to ``'fixmatch'`` in [train.sh](https://github.com/LiheYoung/UniMatch/blob/main/scripts/train.sh).

### Supervised Baseline

Modify the ``method`` from ``'unimatch'`` to ``'supervised'`` in [train.sh](https://github.com/LiheYoung/UniMatch/blob/main/scripts/train.sh), and double the ``batch_size`` in configuration file if you use the same number of GPUs as semi-supervised setting (no need to change ``lr``). 


## Citation

If you find this project useful, please consider citing:

```bibtex
@inproceedings{unimatch,
  title={Revisiting Weak-to-Strong Consistency in Semi-Supervised Semantic Segmentation},
  author={Yang, Lihe and Qi, Lei and Feng, Litong and Zhang, Wayne and Shi, Yinghuan},
  booktitle={CVPR},
  year={2023}
}
```

We have some other works on semi-supervised semantic segmentation:

- [[CVPR 2022] ST++](https://github.com/LiheYoung/ST-PlusPlus) 
- [[CVPR 2023] AugSeg](https://github.com/ZhenZHAO/AugSeg)
- [[CVPR 2023] iMAS](https://github.com/ZhenZHAO/iMAS)
