# Simple federating learning model

This model is the simplest implementation of federated learning model.

## Build docker

```bash
docker build -t <name> <path>
```

## Usage

```bash
docker run --privileged -it <name> bash
horovodrun -np 3 -H localhost:3 python -W ignore  train.py
```

## Approach
I use PyTorch framework and Horovod for parallelism.
I take horovod documentation example as a basis. I change order of training, test loss calculated only on one root model. Models avareges in every backward step by common Horovod optimizer. 
