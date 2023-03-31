# CLIP-fine-tune-example

### Workflow
<img src="https://github.com/ambipomyan/CLIP-fine-tune-example/blob/main/example_01.png" alt= “example_01” width="500">
training: contrastive learning - freeze parameters of the original model -> modify the output layer -> update the modified output layer only

<img src="https://github.com/ambipomyan/CLIP-fine-tune-example/blob/main/example_02.png" alt= “example_02” width="500">
inference: image-caption matching after creating a classifier

### Run
Use virtual environment to execute the script
```
export LD_LIBRARY_PATH=/path/to/envs/clip/lib/:$LD_LIBRARY_PATH
python sample_fine_tune_MNIST.py
```
When using 'RN50' the `loss`:
```
2.2762565135955812
2.0903772354125976
2.0065130233764648
1.917322850227356
1.847741389274597
1.7837785959243775
1.736946702003479
1.6968570470809936
1.678084373474121
1.6762619018554688
1.67263662815094
1.6719923973083497
1.6510158777236938
1.6455634355545044
1.6320341110229493
1.629179835319519
1.6243189573287964
1.6205157995224
1.6189888954162597
1.6184375762939454
...
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
    ...

    train_set = MyDataset("data/MNIST_train_0.txt", preprocess)
    train_loader = DataLoader(train_set, batch_size=10, shuffle=True, num_workers=0)
    
    test_set = MyDataset("data/MNIST_test_0.txt", preprocess)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0)
    
    train(model, device, train_loader, 20)
    
    torch.save(model.state_dict(), 'models/mnistCLIP.pt')
    
    ...
    
    model_ft.load_state_dict(torch.load('models/mnistCLIP.pt'))
    model_ft.eval()
    
    weights = zeroshot_classifier(model_ft, classnames, templates)
    
    test(model_ft, weights, device, test_loader, 1)
```

###### load model
Use `clip.load()` to get the pretrained model, there are multiple models provided, in this example, 4 of them are evaluated.
Besides, the output layer can be modified to the only layer updating parameters, then, parameters are freezed and a fully-connected layer `model.fc` is attached

###### load data
Use `preprocess()` and `clip.tokenize()` to get images and texts, then, the data needs to be loaded of the format: data_path label, then a costomized Dataset is needed

###### train
Use `model(images, texts)` to get logits or use `encode_image()`/`encode_text()` methods to get image/text features, then get logits

###### test
Use all 10 classes as the (zero-shot) classifier

### Results
5x10 image train, 5x10 image test, batch size 10, epoch 20: Compare the accurancy measured by ratio before (zero-shot inference) and after fine-tuning: (original -> fine-tuned)
```
RN50:     56% -> 60%
RN101:    52% -> 68%
ViT-B/16: 66% -> %
ViT-B/32: 52% -> %
```
Use "zero" to replace "0", etc.
```
RN50:     34% -> %
RN101:    36% -> %
ViT-B/16: 54% -> %
ViT-B/32: 22% -> %
```
*These resutls are on CPU, there still some problems for execution on GPU, mainly caused by data type

### References
- https://github.com/openai/CLIP
- https://learn.microsoft.com/en-us/windows/ai/windows-ml/tutorials/pytorch-train-model
- https://github.com/openai/CLIP/issues/164
- https://github.com/openai/CLIP/issues/57
