import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import ast
import json 

class CustomDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        with open(csv_file,"r") as file:
            self.data=json.load(file) 
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data[index]
        image_path = row["data"]["image"].split("-")[-1]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = row["label"]
        label = ast.literal_eval(label)
        points = label[0]["points"]
        x_min, y_min = min([point[0] for point in points]), min([point[1] for point in points])
        x_max, y_max = max([point[0] for point in points]), max([point[1] for point in points])
        label = {"x_min": x_min, "y_min": y_min, "x_max": x_max, "y_max": y_max}
        return image, label


def train_model():
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Define dataset and dataloader
    dataset = CustomDataset("labelj.json", transform=transform)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Define model
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 4)  # 4 outputs for bounding box coordinates

    # Define loss function and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Train model
    num_epochs = 10
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(dataloader):
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, torch.stack([torch.tensor(list(label.values())) for label in labels]))
            loss.backward()
            optimizer.step()

            if (i+1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}")


if __name__ == "__main__":
    train_model()
