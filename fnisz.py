import torchvision
import torch.nn as nn
import torch
import json
from PIL import Image
from torchvision.transforms import Resize
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import torchvision.transforms as T
from tqdm.notebook import tqdm


class TrainingDataset(torch.utils.data.Dataset):
    def __init__(self, data, validation=False):
        self.images = []
        self.boxes = []


        x_final, y_final = 600, 1200
        transform = Resize((x_final, y_final))
        to_tensor = ToTensor()
        for item in data:
            file_name = item['data']['image'].split('-')[-1]

            image_path = file_name  # Zmień odpowiednio ścieżkę do katalogu ze zdjęciami
            image = Image.open(image_path)
            image = to_tensor(transform(image))*255
            image = image.type(torch.uint8)
            self.images.append(image)
            points = item['annotations'][0]['result'][0]['value']['points']

            xmin = min(point[0] for point in points)/100*x_final
            ymin = min(point[1] for point in points)/100*y_final
            xmax = max(point[0] for point in points)/100*x_final
            ymax = max(point[1] for point in points)/100*y_final
            box = torch.tensor([[xmin, ymin, xmax, ymax]])
            self.boxes.append(box)

        self.labels = torch.zeros((len(data), 1), dtype=torch.int64)


        transforms = [#T.ElasticTransform(alpha=45.0), #T.Grayscale(),
                    #   T.ColorJitter(brightness=.5, hue=.3),
                    #   T.RandomInvert(p=0.7), T.RandomPosterize(bits=2),
                    #   T.RandomSolarize(threshold=0.5),
                    #   T.RandomAdjustSharpness(sharpness_factor=2),
                    #   T.RandomAutocontrast(),
                    #   T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
                      T.RandomGrayscale(p=0.5),
                      #T.RandomSolarize(1, p=0.9)



                      ]
        # self.transforms = T.Compose([*[T.RandomApply([transform], p=0.5) for transform in transforms]])
        # if validation:
        #     self.transforms = None
        transforms = transforms * 5

        self.transforms = T.Compose([T.RandomApply(transforms, p=0.5)])

    # def crop_img_with_box(self, img, box):
    #     # TODO: crop image with box
    #     return img, box

    def __getitem__(self, idx):
        out_img = self.images[idx]
        if self.transforms is not None:
            out_img = self.transforms(out_img)

        out_box = self.boxes[idx]
        # out_img, out_box = self.crop_img_with_box(out_img, out_box)
        return out_img/255, out_box, self.labels[idx]

    def __len__(self):
        return len(self.images)
    


model = torchvision.models.detection.fasterrcnn_resnet50_fpn()
model.roi_heads.box_predictor.cls_score = nn.Linear(1024, 1, bias=True)
model.roi_heads.box_predictor.bbox_pred = nn.Linear(1024, 4, bias=True)
model.load_state_dict(torch.load('model.pth'))

#Opcjonalne zamrożenie wag
weights_dict = dict(model.named_parameters())
for k, v in weights_dict.items():
    if "box_predictor" not in k:
        v.requires_grad = False


with open('labelj.json') as f:
    data = json.load(f)


#Hiperparametr
epochs = 20
#lr, momentum - hiperparametry
#można spróbować torch.optim.Adam
# optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
training_dataset = TrainingDataset(data)
#batch_size = 2 - hiperparametr
training_dataloader = torch.utils.data.DataLoader(training_dataset, batch_size=2, shuffle=True)
val_dataset = TrainingDataset(data, validation=True)
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
        print(losses, d['boxes'])
    with torch.no_grad():
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