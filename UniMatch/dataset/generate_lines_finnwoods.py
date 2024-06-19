from transform import map_to_nine


# map_to_nine tar in en bild och returenerar bilden med bara träd och bakgrund.


import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
import cv2

from matplotlib.patches import Rectangle


def find_region_bb(labeled_array, num_features):
    region_corners = []

    for label in range(1, num_features + 1):
        region_slice = ndimage.find_objects(labeled_array == label)[0]
        top_left = (region_slice[0].start, region_slice[1].start)
        bottom_right = (region_slice[0].stop, region_slice[1].stop)
        region_corners.append((top_left, bottom_right))

    return region_corners


def find_region_corners(labeled_array, num_features):
    region_corners = []

    for label in range(1, num_features + 1):
        region_mask = labeled_array == label

        # Find contours of the region
        contours, _ = cv2.findContours(region_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Extract corners of the contours
        corners = np.concatenate(contours[0])  # Combine all contours into a single array
        print(corners)
        # Find the extreme points
        # print("This should be upper right corner: ", max(corners, key=lambda coord: (coord[1], coord[0])))
        # print("This should be the upper left corner: ",  max(corners, key=lambda coord: (coord[1], -coord[0])))
        # print("This should be the lower right corner: ",  max(corners, key=lambda coord: (-coord[1], coord[0])))
        # print("This should be the lower left corner: ",  max(corners, key=lambda coord: (-coord[1], -coord[0])))
        top_left = tuple(max(corners, key=lambda coord: (-coord[0]-50, coord[1])))
        top_right = tuple(max(corners, key=lambda coord: (coord[0]+50, coord[1])))
        bottom_left = tuple(max(corners, key=lambda coord: (-coord[0]-50, -coord[1])))
        bottom_right = tuple(max(corners, key=lambda coord: (coord[0]+50, -coord[1])))

        region_corners.append((top_left, top_right, bottom_left, bottom_right))

    return region_corners


# assign directory
img_directory = 'data/FinnForest/rgb/train'
directory = 'data/FinnForest/annotations/train/semantic'

# iterate over files in
# that directory

# with open('labeled.txt', 'w') as file:
#    for filename in os.listdir(directory):
#        f = os.path.join(directory, filename)
#        # checking if it is a file
#        if os.path.isfile(f):
#            img = Image.open(f)
#            img = np.array(img)
#            img = map_to_nine(img)
#            idx = f.split("/")[5]
#            idx = idx.split(".")[0]
#            file.write("rgb/train/" + str(idx) + ".jpg" + " " + "annotations/train/semantic/" + str(idx) + ".png")
#            file.write('\n')

img_directory = 'data/FinnForest/rgb/val'
directory = 'data/FinnForest/annotations/val/semantic'

# iterate over files in
# that directory


with open('val.txt', 'w') as file:
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        # checking if it is a file
        if os.path.isfile(f):
            img = Image.open(f)
            img = np.array(img)
            img = map_to_nine(img)
            # plt.imshow(img)
            # plt.show()

            labeled_array, num_features = ndimage.label(img)
            print(num_features)

            # Plot the original binary image and the labeled regions
            plt.figure(figsize=(10, 5))

            plt.subplot(1, 2, 1)
            plt.imshow(img, cmap='gray')
            plt.title('Original Binary Image')

            plt.subplot(1, 2, 2)
            plt.imshow(labeled_array, cmap='viridis')  # cmap='viridis' for better visualization of labels
            plt.title('Labeled Regions')

            plt.show()

            # Assuming labeled_array and num_features are already defined
            corners = find_region_corners(labeled_array, num_features)

            # Plot the original binary image and the labeled regions with corners
            plt.figure(figsize=(12, 6))

            # Plot original labeled image
            plt.subplot(1, 2, 1)
            plt.imshow(labeled_array, cmap='viridis')
            plt.title('Labeled Regions')

            # Plot labeled regions with corners
            plt.subplot(1, 2, 2)
            plt.imshow(labeled_array, cmap='viridis')

            for i, (top_left, top_right, bottom_left, bottom_right) in enumerate(corners, 1):
                all_corners = np.array([top_left, top_right, bottom_left, bottom_right])
                #print(all_corners)
                plt.scatter(all_corners[:, 0], all_corners[:, 1], label=f'Region {i}', color='red', s=10)

            plt.title('Labeled Regions with Corners')
            plt.legend()
            plt.show()

            # Assuming labeled_array and num_features are already defined
            corners = find_region_bb(labeled_array, num_features)

            # Plot the original binary image and the labeled regions with corners
            plt.figure(figsize=(12, 6))

            # Plot original binary image
            plt.subplot(1, 2, 1)
            plt.imshow(labeled_array, cmap='viridis')
            plt.title('Labeled Regions')

            # Plot labeled regions with corners
            plt.subplot(1, 2, 2)
            plt.imshow(labeled_array, cmap='viridis')

            for i, (top_left, bottom_right) in enumerate(corners, 1):
                plt.plot([top_left[1], top_left[1], bottom_right[1], bottom_right[1], top_left[1]],
                         [top_left[0], bottom_right[0], bottom_right[0], top_left[0], top_left[0]],
                         label=f'Region {i}', color='red', linewidth=2)

            plt.title('Labeled Regions with Corners')
            plt.legend()
            plt.show()


            idx = f.split("/")[5]
            idx = idx.split(".")[0]
            file.write("rgb/val/" + str(idx) + ".jpg" + " " + "annotations/val/semantic/" + str(idx) + ".png")
            file.write('\n')
