import torch
import torchvision.utils as tvu
from PIL import Image
from torchvision import transforms
convert_tensor = transforms.ToTensor()

image = Image.open("./static/glasses.png")
image_tensor = convert_tensor(image)


a = torch.rand_like(image_tensor) + image_tensor

tvu.save_image(a, "example.png")