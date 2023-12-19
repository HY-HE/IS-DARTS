# Introduction
**IS-DARTS** is accepted by AAAI 2024.

IS-DARTS: Stabilizing DARTS through Precise Measurement on Candidate Importance
Hongyi He, Longjun Liu, Haonan Zhang, Nanning Zheng

**This code is based on the implementation of  [AutoDL-Projects](https://github.com/D-X-Y/AutoDL-Projects).**

## Preparations

Please install Python>=3.6 and PyTorch>=1.5.0.

if you want to search on **NAS-Bench-201**, please install nas-bench-201 api:

```
pip install nas-bench-201
```

Then download the benchmark file from 
[Google Drive](https://drive.google.com/file/d/1SKW0Cu0u8-gb18zDpaAGi0f74UdXeGKs/view) 
to the project folder.

## Usage

###Search on NAS-Bench-201

```
CUDA_VISIBLE_DEVICES=0 bash isdarts_nasbench.sh path_to_data path_to_benchmark_file log_name
```
The evaluation results are given by the benchmark file.

###Search on DARTS search space

####Searching

Search on CIFAR-10:
```
CUDA_VISIBLE_DEVICES=0 bash isdarts_darts.sh cifar10 path_to_data log_name
```
Searching on ImageNet coming soon.

Following AutoDL, if you want to train the searched architecture found by the above scripts, 
you need to add the config of that architecture (will be printed in log) in 
**xautodl/nas_infer_model/DXYs/genotypes.py**

#####Evaluation

Evaluate using one GPU:
```
CUDA_VISIBLE_DEVICES=0 bash basic-main.sh cifar10/cifar100/imagenet-1k path_to_data log_name
```

Evaluate using more than one GPU:
```
CUDA_VISIBLE_DEVICES=0,1,2,3 bash basic-main-distributed.sh cifar10/cifar100/imagenet-1k path_to_data log_name number_of_gpus
```
