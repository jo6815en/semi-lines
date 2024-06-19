from transform import map_to_nine

# map_to_nine tar in en bild och returenerar bilden med bara träd och bakgrund.

import os
import json
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
import cv2


def find_region_bb(labeled_array, num_features):
    region_corners = []

    for label in range(1, num_features + 1):
        region_slice = ndimage.find_objects(labeled_array == label)[0]
        top_left = (region_slice[0].start, region_slice[1].start)
        bottom_right = (region_slice[0].stop, region_slice[1].stop)
        region_corners.append((top_left, bottom_right))

    return region_corners


def find_region(labeled_array, num_features):
    region_corners = []

    for label in range(1, num_features + 1):
        region_mask = labeled_array == label

        # Find contours of the region
        contours, _ = cv2.findContours(region_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Extract corners of the contours
        corners = [tuple(contour.squeeze()) for contour in contours[0]]

        region_corners.append(corners)

    return region_corners


def find_closest_point(corner, points):
    distances = np.linalg.norm(points - corner, axis=1)
    closest_index = np.argmin(distances)
    return points[closest_index]


def get_corners_for_image(f):
    img = Image.open(f)
    img = np.array(img)
    img = map_to_nine(img)
    # print(img.shape)

    labeled_array, num_features = ndimage.label(img)

    bb_corners = find_region_bb(labeled_array, num_features)

    region = find_region(labeled_array, num_features)
    loader = zip(region, bb_corners)

    final_corners = []

    for i, (corners, (top_left, bottom_right)) in enumerate(loader):
        # Find the closest point for each bounding box corner
        bounding_box_corners = np.array([[top_left[1], top_left[0]], [top_left[1], bottom_right[0]],
                                         [bottom_right[1], bottom_right[0]], [bottom_right[1], top_left[0]]])
        points_around_interest = np.array(corners)

        closest_points = [find_closest_point(corner, points_around_interest) for corner in bounding_box_corners]
        final_corners.append(np.array(closest_points))

    final_corners = np.concatenate(final_corners, axis=0)

    # for i in range(len(final_corners)):
    #    point = final_corners[i]
    #    plt.imshow(labeled_array, cmap='viridis')
    #    plt.scatter(point[0], point[1], label=f'Region {i}', color='red', s=10)
    #    plt.show()

    return final_corners


def main():

    # assign directory
    # img_directory = 'data/FinnForest/rgb/train'
    directory = 'data/FinnForest/annotations/train/semantic'

    anno_file = []

    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        # checking if it is a file
        if os.path.isfile(f):
            lines = get_corners_for_image(f)

            lines = [[int(x) for x in point] for point in lines]

            # print(lines)
            lines_list = []
            point_idx = 0
            for i in range(int(len(lines) / 4)):
                x0 = lines[point_idx][0]
                y0 = lines[point_idx][1]
                x1 = lines[point_idx + 1][0]
                y1 = lines[point_idx + 1][1]
                x3 = lines[point_idx + 2][0]
                y3 = lines[point_idx + 2][1]
                x2 = lines[point_idx + 3][0]
                y2 = lines[point_idx + 3][1]
                lines_list.append([x0, y0, x1, y1])
                lines_list.append([x2, y2, x3, y3])
                point_idx = point_idx + 4
            # print(lines_list)
            idx = f.split("/")[5]
            idx = idx.split(".")[0]
            # print(idx)
            anno_file.append({'filename': idx + ".jpg", 'lines': lines_list, 'height': 720, 'width': 1280})

    with open('annos_train_finnwoods.json', 'w') as json_file:
        json.dump(anno_file, json_file, indent=2)

    img_directory = 'data/FinnForest/rgb/val'
    directory = 'data/FinnForest/annotations/val/semantic'

    # iterate over files in
    # that directory

    anno_file = []

    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        # checking if it is a file
        if os.path.isfile(f):
            lines = get_corners_for_image(f)

            lines = [[int(x) for x in point] for point in lines]

            # print(lines)
            lines_list = []
            point_idx = 0
            for i in range(int(len(lines)/4)):
                x0 = lines[point_idx][0]
                y0 = lines[point_idx][1]
                x1 = lines[point_idx+1][0]
                y1 = lines[point_idx+1][1]
                x3 = lines[point_idx+2][0]
                y3 = lines[point_idx+2][1]
                x2 = lines[point_idx+3][0]
                y2 = lines[point_idx+3][1]
                lines_list.append([x0, y0, x1, y1])
                lines_list.append([x2, y2, x3, y3])
                point_idx = point_idx + 4
            # print(lines_list)
            idx = f.split("/")[5]
            idx = idx.split(".")[0]
            # print(idx)
            anno_file.append({'filename': idx + ".jpg", 'lines': lines_list, 'height': 720, 'width': 1280})

    with open('annos_val_finnwoods.json', 'w') as json_file:
        json.dump(anno_file, json_file, indent=2)


if __name__ == '__main__':
    main()
