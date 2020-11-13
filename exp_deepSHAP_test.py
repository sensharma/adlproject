import torch
import shap
import os
import numpy as np
from data.dataloaders import mnist_loader
from models.cnn import Net
from paths import MNIST_CNN_PATH as PATH
# PATH = os.path.abspath('./saved_models/test_exp')

use_cuda = torch.cuda.is_available() 
device = "cuda" if use_cuda else "cpu"


model = Net()
checkpoint = torch.load(f'{PATH}/checkpoint', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)

test_loader = mnist_loader(train=False, test_limit=1005, batch_size=1005)

for batch_idx, (images, target) in enumerate(test_loader):
    images, target = images.to(device), target.to(device)

# background = torch.zeros_like(images).to(device)
# background = np.zeros_like(images)
background = images[:100]
test_images = images[100:105]

e = shap.DeepExplainer(model, background)
shap_values = e.shap_values(test_images)

shap_numpy = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_values]
test_numpy = np.swapaxes(np.swapaxes(test_images.cpu().numpy(), 1, -1), 1, 2)

shap.image_plot(shap_numpy, -test_numpy)
