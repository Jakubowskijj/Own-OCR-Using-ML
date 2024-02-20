import torchvision
import torch.nn as nn
import torch

model = torchvision.models.detection.fasterrcnn_resnet50_fpn()
model.roi_heads.box_predictor.cls_score = nn.Linear(1024, 1, bias=True)
model.roi_heads.box_predictor.bbox_pred = nn.Linear(1024, 4, bias=True)

#Opcjonalne zamrożenie wag
weights_dict = dict(model.named_parameters())
for k, v in weights_dict.items():
    if "box_predictor" not in k:
        v.requires_grad = False
		
		
import json
from PIL import Image
from torchvision.transforms import Resize
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt


# Wczytaj plik JSON
with open('labelj.json') as f:
    data = json.load(f)
	
	
import torchvision.transforms as T
class TrainingDataset(torch.utils.data.Dataset):
    def __init__(self, images, targets):
        self.images = []
        self.boxes = []
        

        for item in data:
            
            file_name = item['data']['image'].split('-')[-1]
    
            image_path = r'C:\Users\Boows\Desktop\sieci\\' + file_name  # Zmień odpowiednio ścieżkę do katalogu ze zdjęciami
            image = Image.open(image_path)
    
            transform = Resize((600, 1200))
            to_tensor = ToTensor()
            image = transform(image)
            image = to_tensor(image)

            self.images.append(image)

            self.boxes = targets
            points = item['annotations'][0]['result'][0]['value']['points']
    
            xmin = min(point[0] for point in points)
            ymin = min(point[1] for point in points)
            xmax = max(point[0] for point in points)
            ymax = max(point[1] for point in points)
            box = torch.tensor([[xmin, ymin, xmax, ymax]])
            self.boxes.append(box)
            
            targets = boxes
        self.labels = torch.zeros((len(images), 1), dtype=torch.int64)


        self.transforms = T.Compose([
        T.RandomApply(T.CenterCrop(10),p=0.5),
        T.RandomApply(T.PILToTensor(),p=0.5),
        T.RandomApply(T.ElasticTransform(alpha=45.0),p=0.5),
        T.RandomApply(T.ConvertImageDtype(torch.float),p=0.5)])

    # def crop_img_with_box(self, img, box):
    #     # TODO: crop image with box
    #     return img, box

    def __getitem__(self, idx):
        out_img = self.images[idx]
        if self.transforms is not None:
            out_img = self.transforms(out_img)

        out_box = self.boxes[idx]
        out_img, out_box = self.crop_img_with_box(out_img, out_box)

        return out_img, out_box, self.labels[idx]

    def __len__(self):
        return len(self.images)

from tqdm.notebook import tqdm

#Hiperparametr
epochs = 10
#lr, momentum - hiperparametry
#można spróbować torch.optim.Adam 
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
training_dataset = TrainingDataset(data)
#batch_size = 2 - hiperparametr
training_dataloader = torch.utils.data.DataLoader(training_dataset, batch_size=2, shuffle=True)
val_dataset = TrainingDataset(data)
val_dataloader = torch.utils.data.DataLoader(training_dataset, batch_size=2, shuffle=True)

for epoch in tqdm(list(range(epochs))):
    for images, boxes, labels in training_dataloader:
        images = list(image.to(device) for image in images)
        targets = []
        for i in range(len(images)):
            d = {}
            d['boxes'] = boxes[i].to(device)
            d['labels'] = labels[i].to(device)
            targets.append(d)
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        losses.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(losses)

    for images, boxes, labels in val_dataloader:
        images = list(image.to(device) for image in images)
        targets = []
        for i in range(len(images)):
            d = {}
            d['boxes'] = boxes[i].to(device)
            d['labels'] = labels[i].to(device)
            targets.append(d)
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        print(f"Val loss: {losses.item()}")

#Zapisanie modelu
torch.save(model.state_dict(), "model.pth")