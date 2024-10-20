# SEMI-LINES

![Header Image](/images/header.png)

This is the official PyTorch implementation of our semi supervised method for line segement detection. 

## Pretrained Weights

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

## Dataset

- Finnwood: [Images](https://github.com/juanb09111/FinnForest)
- Wireframe: [Link to git](https://github.com/huangkuns/wireframe) 
- Spruce A & B:

Annotations from Finnwood are already in data/FinnForest/.

Please modify your dataset path in configuration files.


```
├── data/
    ├── FinnForest/
        ├── rgb/
        ├── annos_train_finnwoods.json
        └── annos_val_finnwoods.json
    ├── sam_segs/
        ├── skrylle_frames/
        ├── snoge_frames/
        ├── annos_snoge.json
        └── annos_skrylle_new.json
    └── Wireframe/
        ├── images/
        ├── train.json
        └── valid.json
```

## Usage

### Docker

### Pull the Docker Image

Pull the Docker image from Docker Hub using the following command:

```bash
docker pull johannaengman/semi-lines:latest
```

This will fetch the latest version of the Docker image.

### Running the Container with Mounted Data

To run the container and mount your local data, use the following command:

```bash
docker run -it --gpus all -v /path/to/your/data:/home2/johannae/semi-lines/UniMatch johannaengman/semi-lines:latest
```

#### Explanation:
- ` /path/to/your/data`: Replace with the directory of your code and data. 
- `-it`: Runs the container interactively.
- `-v /path/to/your/data:`: Mounts your local directory `/path/to/your/data` to the `/home2/johannae/semi-lines/UniMatch` directory inside the container. Replace `/path/to/your/data` with the actual path to your dataset.

## Running the Training Script

Once the container is running and your data is mounted, execute the script inside the container:

```bash
sh scripts/train.sh  <num_gpu> <port>
```

This will run the semi-supervised method with pretraind weights from the supervised method. To run the supervised method in order to obtain the weights, run `scripts/train_supervised.sh`

## Running the Inference Script

Once the container is running and your data is mounted, execute the script inside the container:

```bash
sh scripts/inference.sh  <num_gpu> <port>
```

In the inference script you can change on what method you want to run and with which test dataset.

## Citation

If you find this project useful, please consider citing:

```bibtex

```

