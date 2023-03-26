import torch
import torch.nn as nn
from torch.optim import Adam
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import clip
from PIL import Image
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

model_pre, preprocess = clip.load("ViT-B/32", device=device)

class MyDataset(Dataset):
    def __init__(self, data_path):
        f = open(data_path, 'r')
        data = []
        for line in f:
            line = line.rstrip()
            words = line.split()
            data.append((words[0], words[1]))
        self.data = data
        
    def __getitem__(self, index):
        i, t = self.data[index]
        
        image = preprocess(Image.open(i)).unsqueeze(0)
        text = clip.tokenize(t)
        
        return image, text
    
    def __len__(self):
        return len(self.data)

def get_model(model_pre, device):
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
    
    
    return model

def train(model, device, train_loader, epochs):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(model.fc.parameters(), lr=0.001, weight_decay=0.0001)

    model.to(device)

    model.train()

    for epoch in range(epochs):
        total_loss = 0.0
        for idx, data in enumerate(train_loader):
            images, texts = data
            images = Variable(images.squeeze(1).to(device))
            texts = Variable(texts.squeeze(1).to(device))
            
            optimizer.zero_grad()
            
            logits_images, logits_texts = model(images, texts)
            
            loss = loss_fn(logits_images, logits_texts)
            
            loss.requires_grad_(True)
            loss.backward()
            
            optimizer.step()
            
            total_loss += loss.item()
            avg_loss = total_loss / len(train_loader)
            

def test(model, device, test_loader, epochs):
    counts = 0
    with torch.no_grad():
        for idx, data in enumerate(test_loader):
            images, texts = data
            images = Variable(images.squeeze(1).to(device))
            texts = Variable(clip.tokenize(["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]).to(device))
            
            logits_per_image, logits_per_text = model(images, texts)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()
            
            print(np.argmin(probs[0]))
            if np.argmin(probs[0]) > (int(idx/5)-1) and np.argmin(probs[0]) <= int(idx/5):
                counts += 1
    
    print(counts*2)

def main():
    model = get_model(model_pre, device)
    
    train_set = MyDataset("data/MNIST_train.txt")
    train_loader = DataLoader(train_set, batch_size=50, shuffle=True, num_workers=0)
    
    test_set = MyDataset("data/MNIST_test.txt")
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0)
    
    train(model, device, train_loader, 20)
    test(model, device, test_loader, 1)

if __name__ == '__main__':
    main()
