# CLIP-fine-tune-example

### Run
Use virtual environment to execute the script
```
python sample_fine_tune_MNIST.py
```

### Data
Use the .txt file of the format below: `data_path label`
```
data/MNIST/train/0/3.jpg 0
data/MNIST/train/0/10.jpg 0
data/MNIST/train/0/13.jpg 0
data/MNIST/train/0/25.jpg 0
data/MNIST/train/0/28.jpg 0
...
```
In this example, both train and test data are of size 5 x 10(classes), and the data is strutured as follow:
```
-- sample_fine_tune_MNIST.py
-- data
   |-- MNIST_train.txt
   |-- MNIST_test.txt
   `-- MNIST
       |-- train
       |   |-- 0
       |   |   |-- aaaa.jpg
       |   |   |-- bbbb.jpg
       |   |   |   ...
       |   |   `-- zzzz.jpg
       ...
       `-- test
           |-- 0
           |   |-- AAAA.jpg
           |   |-- BBBB.jpg
           |   |   ...
           |   `-- ZZZZ.jpg
           ...
```

### Implementation
###### overall
Define a `main` method
```
def main():
    train_set = MyDataset("data/MNIST_train.txt")
    train_loader = DataLoader(train_set, batch_size=50, shuffle=True, num_workers=0)
    
    test_set = MyDataset("data/MNIST_test.txt")
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0)
    
    train(model, device, train_loader, 20)
    test(model, device, test_loader, 1)
```
###### load model
Use `clip.load()` to get the pretrained model, there are multiple models provided, in this example, 4 of them are evaluated.
Besides, the output layer can be modified to the only layer updating parameters, then, parameters are freezed and a fully-connected layer `model.fc` is attached

###### load data
Use `preprocess()` and `clip.tokenize()` to get images and texts, then, the data needs to be loaded of the format: data_path label, then a costomized Dataset is needed

###### train
Use `model(images, texts)` to get logits

###### test
Use all 10 classes as the texts

### Results
5x10 image train, 5x10 image test, batch size 10, epoch 20
Compare the accurancy measured by ratio before (zero-shot inference) and after fine-tuning: (original -> modifying model -> modifying preprocess)
```
RN50:      6% ->  4% -> 4%
RN101:     0% -> 10% -> 0%
ViT-B/16:  0% ->  0% -> 4%
ViT-B/32: 10% -> 10% -> 10%
```
Use "zero" to replace "0", etc.
```
RN50:      8% ->  4% -> 2%
RN101:     2% -> 10% -> 2%
ViT-B/16:  4% ->  8% -> 4%
ViT-B/32:  2% ->  2% -> 4%
```

### References
- https://github.com/openai/CLIP
- https://learn.microsoft.com/en-us/windows/ai/windows-ml/tutorials/pytorch-train-model
