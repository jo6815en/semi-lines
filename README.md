# SEMI-LINES

![Header Image](/images/header.png)

This is the official PyTorch implementation of our semi supervised method for line segement detection. 

### Pretrained Weights

There are some pretrained weights for the two datasets and some splits.

```
├── ./exp/
    ├── finnswoods/
        ├── supervised/
        └── semisup/
    └── wireframe/
        ├── supervised/
        └── semisup/
     
```

### Dataset

- FInnwood: [JPEGImages](https://github.com/juanb09111/FinnForest)
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

### Running the Script

Once the container is running and your data is mounted, execute the script inside the container:

```bash
sh scripts/train.sh  <num_gpu> <port>
```

This will run the semi-supervised method with pretraind weights from the supervised method. To run the supervised method in order to obtain the weights, run `scripts/train_supervised.sh`



## Citation

If you find this project useful, please consider citing:

```bibtex

```

