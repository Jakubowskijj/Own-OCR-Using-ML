import torchvision
import torch.nn as nn
import torch
import json
from PIL import Image
from torchvision.transforms import Resize, ToTensor
import torchvision.transforms as T
from tqdm.notebook import tqdm

# Definicja klasy TrainingDataset
class TrainingDataset(torch.utils.data.Dataset):
    def __init__(self, data, image_dir):
        self.images = []
        self.boxes = []
        self.labels = []

        for item in data:
            file_name = item['data']['image'].split('-')[-1]
            image_path = image_dir + file_name
            image = Image.open(image_path)
            transform = Resize((600, 1200))
            to_tensor = ToTensor()
            image = transform(image)
            image = to_tensor(image)
            self.images.append(image)

            points = item['annotations'][0]['result'][0]['value']['points']
            xmin = min(point[0] for point in points)
            ymin = min(point[1] for point in points)
            xmax = max(point[0] for point in points)
            ymax = max(point[1] for point in points)
            box = torch.tensor([[xmin, ymin, xmax, ymax]])
            self.boxes.append(box)

            self.labels.append(1)  # Ustawienie etykiety jako 1 dla kartki (można dostosować dla innych etykiet)

        self.transforms = T.Compose([
            T.RandomApply(T.CenterCrop(10), p=0.5),
            T.RandomApply(T.PILToTensor(), p=0.5),
            T.RandomApply(T.ElasticTransform(alpha=45.0), p=0.5),
            T.RandomApply(T.ConvertImageDtype(torch.float), p=0.5)
        ])

    def crop_img_with_box(self, img, box):
        # TODO: crop image with box
        return img, box

    def __getitem__(self, idx):
        out_img = self.images[idx]
        if self.transforms is not None:
            out_img = self.transforms(out_img)

        out_box = self.boxes[idx]
        out_img, out_box = self.crop_img_with_box(out_img, out_box)

        return out_img, out_box, self.labels[idx]

    def __len__(self):
        return len(self.images)


# Tworzenie modelu
model = torchvision.models.detection.fasterrcnn_resnet50_fpn()
model.roi_heads.box_predictor.cls_score = nn.Linear(1024, 1, bias=True)
model.roi_heads.box_predictor.bbox_pred = nn.Linear(1024, 4, bias=True)

# Zamrożenie wag
weights_dict = dict(model.named_parameters())
for k, v in weights_dict.items():
    if "box_predictor" not in k:
        v.requires_grad = False

# Wczytanie danych z pliku JSON
with open('labelj.json') as f:
    data = json.load(f)

# Ścieżka do katalogu ze zdjęciami
image_dir = r'C:\Users\Boows\Desktop\sieci\\'  # Zmień odpowiednio ścieżkę do katalogu ze zdjęciami

# Tworzenie datasetów i dataloaderów
training_dataset = TrainingDataset(data, image_dir)
training_dataloader = torch.utils.data.DataLoader(training_dataset, batch_size=2, shuffle=True)
val_dataset = TrainingDataset(data, image_dir)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=2, shuffle=True)

# Hiperparametry
epochs = 10
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# Trening modelu
for epoch in tqdm(list(range(epochs))):
    model.train()
    total_loss = 0

    for images, boxes, labels in training_dataloader:
        images = list(image.to(device) for image in images)
        targets = [{'boxes': box.to(device), 'labels': label.to(device)} for box, label in zip(boxes, labels)]

        optimizer.zero_grad()
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        losses.backward()
        optimizer.step()

        total_loss += losses.item()

    avg_loss = total_loss / len(training_dataloader)
    print(f"Epoch {epoch+1}: Training loss: {avg_loss}")

    model.eval()
    val_loss = 0

    for images, boxes, labels in val_dataloader:
        images = list(image.to(device) for image in images)
        targets = [{'boxes': box.to(device), 'labels': label.to(device)} for box, label in zip(boxes, labels)]

        with torch.no_grad():
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        val_loss += losses.item()

    avg_val_loss = val_loss / len(val_dataloader)
    print(f"Epoch {epoch+1}: Validation loss: {avg_val_loss}")

# Zapisanie modelu
torch.save(model.state_dict(), "model.pth")
