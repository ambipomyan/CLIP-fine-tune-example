# CLIP-fine-tune-example

##### Run
using virtual environment to execute the script
```
python sample_fine_tune_MNIST.python
```

##### Design
Use 5 x 10 samples in train dataset of MNIST to fine-tune the CLIP   
Use 5 x 10 samples in test dataset of MNIST to test the CLIP   
Compare the accurancy measured by ratio before (zero-shot inference) and after fine-tuning

##### Data
Use the .txt file of the format below: `data_path label`
```
data/MNIST/train/0/3.jpg 0
data/MNIST/train/0/10.jpg 0
data/MNIST/train/0/13.jpg 0
data/MNIST/train/0/25.jpg 0
data/MNIST/train/0/28.jpg 0
...
```

##### Results
Before and after fine-tuning:
```
RN50:      6% ->  4%
RN101:     0% -> 10%
ViT-B/16:  0% ->  0%
ViT-B/32: 10% -> 10%
```
