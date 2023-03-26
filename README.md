# CLIP-fine-tune-example

### Run
Use virtual environment to execute the script
```
python sample_fine_tune_MNIST.python
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
-- MNIST
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
    model = get_model(model_pre, device)
    
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
Compare the accurancy measured by ratio before (zero-shot inference) and after fine-tuning:
```
RN50:      6% ->  4%
RN101:     0% -> 10%
ViT-B/16:  0% ->  0%
ViT-B/32: 10% -> 10%
```
### References
- https://github.com/openai/CLIP
- https://learn.microsoft.com/en-us/windows/ai/windows-ml/tutorials/pytorch-train-model
