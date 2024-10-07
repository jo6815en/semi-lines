import os
from PIL import Image
import numpy as np

def get_unique_colors(image_path):
    img = Image.open(image_path)
    arr = np.array(img)
    flat_arr = arr.reshape(-1, 3)
    unique_colors = np.unique(flat_arr, axis=0)

    return set(map(tuple, unique_colors))

def find_nine_colors(folder):
    unique_colors = set()
    for filename in os.listdir(folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):  # Add other file types if needed
            image_path = os.path.join(folder, filename)
            image_colors = get_unique_colors(image_path)
            unique_colors.update(image_colors)

            if len(unique_colors) >= 9:
                break

    return list(unique_colors)[:9]

# Test the function
folder_path = 'data/FinnForest/annotations/train/semantic'
colors = find_nine_colors(folder_path)
for color in colors:
    print(color)