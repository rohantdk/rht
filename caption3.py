from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import requests
import torch
import matplotlib.pyplot as plt

image = Image.open("C:/Users/tidke/Downloads/img.jpeg").convert('RGB')

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

inputs = processor(images=image, return_tensors="pt")
out = model.generate(**inputs, max_length=50)
caption = processor.decode(out[0], skip_special_tokens=True)

print("Generated Caption:", caption)

plt.imshow(image)
plt.title(caption)
plt.axis('off')
plt.show()