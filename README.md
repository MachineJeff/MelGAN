# MelGAN - TensorFlow Implementation

TF version : tf1.10

## How to train:

### Step 1:

Download your interested audio dataset to "base_dir"(just treat it to be a folder storing datasets model or anything related to melgan), in this project, defalt is Baker-5000 dataset.

### Step 2:

Run **install.sh** for installing some requirement packages.

```shell
./install.sh
```

### Step 3:

Run **preprocess.py** for extracting mel spectrum features from audio datasets. You need to specify the "base_dir" address and which dataset you like to train. 

It will output a file called "melgan_train" which stores the index of mel spectrum features and its corresponding audio datasets. 

You can find more details from the main function in preprocess.py.

```shell
python preprocess.py -b '/data/baker' --dataset 'Baker-5000'
```

### Step 4:

Run **train.py**. You need to specify the "base_dir" and "log_dir" address, "log_dir" will save tensorflow model files ending with ckpt.

```shell
nphup python train.py -b '/data/baker' -l 'melgan_log' >> log &
```



## How to test:

Run **eval.py**. You need to specify the file of mel spectrum feature and the checkpoint. It will output a audio file ending with wav.

```shell
python eval.py -m 'mel-001000.npy' -c 'melgan_model.ckpt-10000'
```



## What else:

1. **freeze2pb.py** is for converting ckpt file to pb file which is helpful in deployment.

2. **pb2inference.py** is a test script used to detect whether the pb model works.




