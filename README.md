# CLIP-fine-tune-example

##### run
using virtual environment to execute the script
```
python sample_fine_tune_MNIST.python
```

##### design
use 5 x 10 samples in train dataset of MNIST to fine-tune the CLIP   
use 5 x 10 samples in test dataset of MNIST to test the CLIP   
compare the accurancy measured by ratio before (zero-shot inference) and after fine-tuning

##### results
before and after fine-tuning:
```
RN50:      6% ->  4%
RN101:     0% -> 10%
ViT-B/16:  0% ->  0%
ViT-B/32: 10% -> 10%
```
