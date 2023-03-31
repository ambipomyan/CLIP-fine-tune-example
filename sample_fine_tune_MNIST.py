import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import clip
from PIL import Image
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

model_pre, preprocess_pre = clip.load('RN50', device=device)

# modifying model structure
model = model_pre
    
for param in model.parameters():
    param.requires_grad = False
    
model.fc = nn.Sequential(
    nn.Flatten(),
    nn.BatchNorm1d(4096),
    nn.Dropout(0.5),
    nn.Linear(4096, 512),
    nn.ReLU(),
    nn.BatchNorm1d(512),
    nn.Linear(512, 10),
    nn.LogSoftmax(dim=1)
)

# modifying preprocess
preprocess = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])

classnames = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ]
templates = ['a photo of the number: "{}".', ]

class MyDataset(Dataset):
    def __init__(self, data_path, preprocess):
        f = open(data_path, 'r')
        data = []
        for line in f:
            line = line.rstrip()
            words = line.split()
            data.append((words[0], words[1]))
        self.data = data
        
    def __getitem__(self, index):
        i, t = self.data[index]
        
        image = preprocess(Image.open(i).convert('RGB')).unsqueeze(0)
        text = clip.tokenize(t)
        
        return image, text
    
    def __len__(self):
        return len(self.data)


def per_accurancy(probs):
    m = max(probs)
    for i, j in enumerate(probs):
    	if j == m:
    	    return i
    

def zeroshot_classifier(classnames, templates):
    with torch.no_grad():
        weights = []
        for classname in classnames:
            texts = [template.format(classname) for template in templates]
            texts = clip.tokenize(texts).to(device)
            text_features = model.encode_text(texts)
            class_embeddings = text_features / text_features.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            weights.append(class_embedding)

        weights = torch.stack(weights, dim=1).to(device)
     
        return weights


def train(model, device, train_loader, epochs):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(model.fc.parameters(), lr=0.001, weight_decay=0.0001)

    model.to(device)

    for epoch in range(epochs):
        total_loss = 0.0
        for idx, data in enumerate(train_loader):
            images, texts = data
            
            images = images.squeeze(1).to(device)
            texts = texts.squeeze(1).to(device)
            
            optimizer.zero_grad()
            
            logits_images, logits_texts = model(images, texts)
            
            loss = loss_fn(logits_images, logits_texts)
            
            loss.requires_grad_(True)
            loss.backward()
            
            optimizer.step()
            
            total_loss += loss.item()
            avg_loss = total_loss / len(train_loader)
            

def test(model, weights, device, test_loader, epochs):
    counts = 0
    with torch.no_grad():
        for idx, data in enumerate(test_loader):
            images, texts = data
            
            images = images.squeeze(1).to(device) 
            
            image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            
            logits = 100 * image_features @ weights
            probs = logits.softmax(dim=-1).cpu().numpy()
            
            #print(per_accurancy(probs[0]))
            if per_accurancy(probs[0]) > (int(idx/5)-1) and per_accurancy(probs[0]) <= int(idx/5):
                counts += 1
    
    print(counts*2)


def main():
    # already get the model and preprocess

    train_set = MyDataset("data/MNIST_train_0.txt", preprocess)
    train_loader = DataLoader(train_set, batch_size=10, shuffle=True, num_workers=0)
    
    test_set = MyDataset("data/MNIST_test_0.txt", preprocess)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0)
    
    train(model, device, train_loader, 20)
    
    weights = zeroshot_classifier(classnames, templates)
    test(model, weights, device, test_loader, 1)

if __name__ == '__main__':
    main()
