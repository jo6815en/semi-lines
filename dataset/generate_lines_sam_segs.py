import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import json
import os

# Specify the path to your .npy file
npy_file_path = '../data/sam_segs/lines_snoge_frames2/im_841_linepoints.npy'
image_file_path = '../data/sam_segs/snoge_frames2/841.png'

# Use np.load() to load the data from the .npy file
points = np.load(npy_file_path)

print(points.shape)
# Load and display the image using Pillow (PIL)
image = Image.open(image_file_path)
plt.imshow(image)

# Plot the points on the image using scatter plot
for i in range(points.shape[2]):
    plt.scatter(points[0, 0, i], points[1, 0, i], marker='.', c='red')
    plt.scatter(points[2, 0, i], points[3, 0, i], marker='.', c='red')
    plt.scatter(points[0, 1, i], points[1, 1, i], marker='.', c='red')
    plt.scatter(points[2, 1, i], points[3, 1, i], marker='.', c='red')

# Customize the plot as needed
plt.title('Points on Image')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()


#directory = '../data/sam_segs/lines_snoge_frames2'
directory = '../data/sam_segs/lines_skrylle_frames'

anno_file = []

for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    # checking if it is a file
    if os.path.isfile(f):
        points = np.load(f)
        if points.shape[0] == 4:
            lines_list = []
            point_idx = 0
            for i in range(points.shape[2]):
                x0 = points[0, 0, i]
                y0 = points[1, 0, i]
                x1 = points[2, 0, i]
                y1 = points[3, 0, i]
                x3 = points[0, 1, i]
                y3 = points[1, 1, i]
                x2 = points[2, 1, i]
                y2 = points[3, 1, i]
                lines_list.append([x0, y0, x1, y1])
                lines_list.append([x2, y2, x3, y3])
            idx = f.split("/")[4]
            idx = idx.split(".")[0]
            idx = idx.split("_")[1]
            anno_file.append({'filename': "image_" + idx + ".png", 'lines': lines_list, 'height': 720, 'width': 1280})
        else:
            print(f)
            print(points.shape)

with open('annos_skrylle_new.json', 'w') as json_file:
    json.dump(anno_file, json_file, indent=2)

