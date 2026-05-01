from pathlib import Path
import json

import matplotlib.pyplot as plt
import requests
import torch
from torchvision.io import read_image
from torchvision.models import EfficientNet_B3_Weights, efficientnet_b3


BASE_DIR = Path(__file__).resolve().parent
IMAGES_DIR = BASE_DIR / "images"


def plot_images_and_embeddings(images, embeddings, filenames):
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    for idx in range(2):
        axs[idx, 0].imshow(images[idx].numpy().transpose(1, 2, 0))
        axs[idx, 0].set_title(f"Imagen original: {filenames[idx]}")
        axs[idx, 0].axis("off")

        embedding_image = embeddings[idx].reshape((48, 32))
        axs[idx, 1].imshow(embedding_image, cmap="viridis")
        axs[idx, 1].set_title(f"Embedding {filenames[idx]}")
        axs[idx, 1].axis("off")

    plt.tight_layout()
    plt.show()


labels_url = (
    "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
)
response = requests.get(labels_url, timeout=30)
response.raise_for_status()
imagenet_labels = json.loads(response.text)

weights = EfficientNet_B3_Weights.DEFAULT
preprocess = weights.transforms()
model = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
model.eval()

imagenes = ["cat.7.jpg", "dog.4.jpg"]
images_processed = []
embeddings = []

for imagen in imagenes:
    image_path = IMAGES_DIR / imagen
    i = read_image(str(image_path))

    with torch.no_grad():
        processed_image = preprocess(i).unsqueeze(0)
        images_processed.append(i)

        outputs = model(processed_image)
        _, predicted_idx = torch.max(outputs, 1)
        predicted_label = imagenet_labels[predicted_idx.item()]
        print(f"Clase predicha por el modelo para {imagen}: {predicted_label}")

        features = model.features(processed_image)
        pooled_features = model.avgpool(features)
        flattened = torch.flatten(pooled_features, 1).detach().cpu().numpy()
        embeddings.append(flattened[0])

plot_images_and_embeddings(images_processed, embeddings, imagenes)
