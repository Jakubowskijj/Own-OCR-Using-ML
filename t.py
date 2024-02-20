import torch
from PIL import Image
from torchvision.transforms import Resize, ToTensor
import torchvision.transforms as T
import matplotlib.pyplot as plt
import torchvision
import torch.nn as nn



model = torchvision.models.detection.fasterrcnn_resnet50_fpn()
model.roi_heads.box_predictor.cls_score = nn.Linear(1024, 1, bias=True)
model.roi_heads.box_predictor.bbox_pred = nn.Linear(1024, 4, bias=True)




# Ścieżka do zapisanego modelu
model_path = r"C:\Users\Boows\Desktop\sieci\model.pth"
model.load_state_dict(torch.load(model_path))



# Wczytanie modelu
model = torchvision.models.detection.fasterrcnn_resnet50_fpn()
model.roi_heads.box_predictor.cls_score = nn.Linear(1024, 1, bias=True)
model.roi_heads.box_predictor.bbox_pred = nn.Linear(1024, 4, bias=True)
model.load_state_dict(torch.load(model_path))
model.eval()

# Przygotowanie transformacji dla testowego zdjęcia
x_final, y_final = 600, 1200
transform = Resize((x_final, y_final))
to_tensor = ToTensor()

# Wczytanie testowego zdjęcia
test_image_path = r"C:\Users\Boows\Desktop\sieci\test3.jpg"
test_image = Image.open(test_image_path)
test_image = to_tensor(transform(test_image))



# Przygotowanie tensora batcha z pojedynczym zdjęciem
test_image_batch = test_image.unsqueeze(0)

# Przesunięcie tensora batcha na odpowiednie urządzenie (GPU, jeśli dostępne)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
test_image_batch = test_image_batch.to(device)

print(test_image_batch.shape)

# Uruchomienie predykcji na modelu
with torch.no_grad():
    predictions = model(test_image_batch)

# Wyświetlenie wyników predykcji
boxes = predictions[0]['boxes'].cpu().numpy()
scores = predictions[0]['scores'].cpu().numpy()
labels = predictions[0]['labels'].cpu().numpy()

# Wyświetlenie testowego zdjęcia z zaznaczonymi bounding boxami i etykietami
fig, ax = plt.subplots(1)
ax.imshow(test_image.permute(1, 2, 0))

for box, score, label in zip(boxes, scores, labels):
    if score > 0.5:  # Wybierz tylko bounding boxy z wynikiem powyżej 0.5
        xmin, ymin, xmax, ymax = box
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color='red'))
        ax.text(xmin, ymin, f"Label: {label}, Score: {score:.2f}", color='red')

plt.axis('off')
plt.show()
