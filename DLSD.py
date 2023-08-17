
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap as lsc
import torch
import argparse

from PIL import Image, ImageDraw
import svgwrite as svg
from deeplsd.utils.tensor import batch_to_device
from deeplsd.models.deeplsd_inference import DeepLSD
from deeplsd.geometry.viz_2d import plot_images, plot_lines

def process_image_with_deeplsd(img, out_folder, device='cuda'):
    # Model config
    conf = {
        'detect_lines': True,
        'line_detection_params': {
            'merge': True,
            'filtering': True,
            'grad_thresh': 3.5,
            'grad_nfa': True,
        }
    }
    # Load the model
    ckpt = 'DeepLSD/weights/deeplsd_md.tar'  # Adjust the checkpoint path
    ckpt = torch.load(str(ckpt), map_location='cpu')
    net = DeepLSD(conf)
    net.load_state_dict(ckpt['model'])
    net = net.to(device).eval()

    # Detect lines
    inputs = {'image': torch.tensor(img, dtype=torch.float, device=device)[None, None] / 255.}
    with torch.no_grad():
        out = net(inputs)
        pred_lines = out['lines'][0]

    return img, out, pred_lines


def remove_overlapping_lines(lines, over_threshold=30):
    # Convert lines to a list of tuples for easier processing
    lines_list = [(tuple(line[0]), tuple(line[1])) for line in lines]
    non_overlapping_lines = []

    for line in lines_list:
        line_overlap = False

        for existing_line in non_overlapping_lines:
            # Compute the distance between the endpoints of the lines
            dist_start = np.linalg.norm(np.array(line[0]) - np.array(existing_line[0]))
            dist_end = np.linalg.norm(np.array(line[1]) - np.array(existing_line[1]))

            # If the distance is less than the threshold, consider them overlapping
            if dist_start < over_threshold or dist_end < over_threshold:
                line_overlap = True
                break

        # If the line doesn't overlap with any existing lines, add it to the list
        if not line_overlap:
            non_overlapping_lines.append(line)

    # Convert the non-overlapping lines back to the original format
    non_overlapping_lines = np.array([list(line) for line in non_overlapping_lines])

    return non_overlapping_lines


def calculate_angle(line1, line2):
    vector1 = np.array(line1[1]) - np.array(line1[0])
    vector2 = np.array(line2[1]) - np.array(line2[0])
    dot_product = np.dot(vector1, vector2)
    norm_product = np.linalg.norm(vector1) * np.linalg.norm(vector2)
    angle_rad = np.arccos(dot_product / norm_product)
    angle_deg = np.degrees(angle_rad)
    return angle_deg


def calculate_length(line):
    x_diff = abs(line[0][0] - line[1][0])
    y_diff = abs(line[0][1] - line[1][1])
    length = np.sqrt(x_diff**2 + y_diff**2)
    return length


def calculate_distance(line1, line2):
    x_diff = abs(line1[0][0] - line2[0][0])
    y_diff = abs(line1[0][1] - line2[0][1])
    dist = np.sqrt(x_diff**2 + y_diff**2)
    return dist


def calculate_point_distance(point, line):
    x0 = point[0]
    y0 = point[1]
    x1 = line[0][0]
    y1 = line[0][1]
    x2 = line[1][0]
    y2 = line[1][1]

    # Calculate the coefficients A, B, and C of the line equation Ax + By + C = 0
    A = y2 - y1
    B = x1 - x2
    C = x2 * y1 - x1 * y2

    # Calculate the distance using the formula
    distance = abs(A * x0 + B * y0 + C) / np.sqrt(A**2 + B**2)
    return distance


def merge_lines_within_distance(lines, merge_threshold=50, orientation_threshold=5):
    merged_lines = []
    remaining_lines = lines.copy()

    while len(remaining_lines) > 0:
        current_line = remaining_lines[0]
        current_merged_line = current_line
        remaining_lines = remaining_lines[1:]

        for index, line in enumerate(remaining_lines):
            start_end_dist = np.linalg.norm(np.array(current_line[0]) - np.array(line[1]))
            end_start_dist = np.linalg.norm(np.array(current_line[1]) - np.array(line[0]))

            if start_end_dist < merge_threshold or end_start_dist < merge_threshold:
                angle = calculate_angle(current_line, line)
                if angle < orientation_threshold:
                    current_merged_line = (
                        (min(current_line[0][0], line[0][0]), min(current_line[0][1], line[0][1])),
                        (max(current_line[1][0], line[1][0]), max(current_line[1][1], line[1][1]))
                    )
                    current_line = current_merged_line
                    remaining_lines[index] = current_merged_line

        # Append the merged line to the list
        merged_lines.append(current_merged_line)
        
        # Remove merged lines by creating a new list without them
        remaining_lines = [line for line in remaining_lines if not np.array_equal(line, current_line)]

    merged_lines = np.array([list(line) for line in merged_lines])
    return merged_lines


def remove_short_parallel_lines(lines, dist_threshold=10):
    # Convert lines to a list of tuples for easier processing
    lines_list = [(tuple(line[0]), tuple(line[1])) for line in lines]
    # segment the lines into horizontal and vertical lines
    horizontal_lines = []
    vertical_lines = []
    filtered_line = []
    remaining_lines = lines_list.copy()
    close_lines = False

    for line in remaining_lines:
        angle = np.arctan((line[1][1] - line[0][1]) / (line[1][0] - line[0][0]))
        angle = angle * 180 / np.pi

        if -45 <= angle <= 45:
            horizontal_lines.append(line)
        else:
            vertical_lines.append(line)

    while len(horizontal_lines) > 0:
        current_line = horizontal_lines[0]
        horizontal_lines = horizontal_lines[1:]
        close_lines = False

        for index, line in enumerate(horizontal_lines):
            # y diff
            dist = abs(line[0][1] - current_line[0][1])

            if dist < dist_threshold:
                close_lines = True

        if not close_lines:
            filtered_line.append(current_line)

    while len(vertical_lines) > 0:
        current_line = vertical_lines[0]
        vertical_lines = vertical_lines[1:]
        close_lines = False

        for index, line in enumerate(vertical_lines):
            # x diff
            dist = abs(line[0][0] - current_line[0][0])

            if dist < dist_threshold:
                close_lines = True

        if not close_lines:
            filtered_line.append(current_line)

    filtered_line = np.array([list(line) for line in filtered_line])
    return filtered_line
    

def extend_all_lines(lines, extension_distance=50):
    # Convert lines to a list of tuples for easier processing
    lines_list = [(tuple(line[0]), tuple(line[1])) for line in lines]
    remaining_lines = lines_list.copy()
    extended_lines = []
    horizontal_lines = []
    vertical_lines = []
    length_temp = 0

    for line in remaining_lines:
        angle = np.arctan((line[1][1] - line[0][1]) / (line[1][0] - line[0][0]))
        angle = angle * 180 / np.pi

        if -45 <= angle <= 45:
            horizontal_lines.append(line)
        else:
            vertical_lines.append(line)

    for hline in horizontal_lines:
        for vline in vertical_lines:
            start_dist = calculate_point_distance(hline[0], vline)
            end_dist = calculate_point_distance(hline[1], vline)

            if start_dist < extension_distance:
                temp_line = ((vline[0][0], hline[0][1]), hline[1])
                length_temp = calculate_length(temp_line)

            elif end_dist < extension_distance:
                temp_line = (hline[0], (vline[0][0], hline[1][1]))
                length_temp = calculate_length(temp_line)
    

        if length_temp > calculate_length(hline):
            extended_lines.append(temp_line)
        else:
            extended_lines.append(hline)

    length_temp = 0

    for vline in vertical_lines:
        for hline in horizontal_lines:
            start_dist = calculate_point_distance(vline[0], hline)
            end_dist = calculate_point_distance(vline[1], hline)

            if start_dist < extension_distance:
                temp_line = ((vline[0][0], hline[0][1]), vline[1])
                length_temp = calculate_length(temp_line)
    
            elif end_dist < extension_distance:
                temp_line = (vline[0], (vline[0][0], hline[1][1]))
                length_temp = calculate_length(temp_line)
    
        if length_temp > calculate_length(vline):
            extended_lines.append(temp_line)
        else:
            extended_lines.append(vline)

    extended_lines = np.array([list(line) for line in extended_lines])
    return extended_lines


def save_image_with_lines(img, pred_lines, out_path):
    plot_images([img], cmaps='gray')
    plot_lines([pred_lines], indices=range(1))
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
    plt.close()


def save_svg(pred_lines,img, svg_path):
    print("Saving svg")

    fig, ax = plt.subplots(figsize=(img.shape[1] / 100, img.shape[0] / 100))
    horizontal_lines = []

    for line in pred_lines:
        angle = np.arctan2((line[1][1] - line[0][1]), (line[1][0] - line[0][0]))
        angle = angle * 180 / np.pi
        if -45 <= angle <= 45:
            horizontal_lines.append(line)

    horizontal_lines.sort(key=lambda x: x[0][1])

    for i in range(len(horizontal_lines) - 1):
        line1 = horizontal_lines[i]
        line2 = horizontal_lines[i + 1]

        # Calculate rectangle coordinates and size
        rect_x = min(line1[0][0], line2[0][0]) + 5
        rect_y = line1[0][1] + 5
        rect_width = abs(line1[1][0] - line1[0][0]) - 5
        rect_height = line2[0][1] - line1[0][1] - 5

        # Draw rectangle between lines
        rect = plt.Rectangle((rect_x, rect_y), rect_width, rect_height, edgecolor='blue', facecolor='blue', alpha=0.2)
        ax.add_patch(rect)

    # Draw lines
    for line in pred_lines:
        start_x, start_y = line[0]
        end_x, end_y = line[1]
        ax.plot([start_x, end_x], [start_y, end_y], color='black', linewidth=2)

    # Save the SVG file
    plt.axis('off')
    plt.gca().invert_yaxis()
    plt.savefig(svg_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def save_png(pred_lines,img, png_path):
    # fill the image with white color
    img = np.full_like(img, 255)

    # draw the lines on the image using ImageDraw and Image from PIL
    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)
    
    # separate the lines into horizontal and vertical lines
    horizontal_lines = []

    for line in pred_lines:
        angle = np.arctan2((line[1][1] - line[0][1]), (line[1][0] - line[0][0]))
        angle = angle * 180 / np.pi

        if -45 <= angle <= 45:
            horizontal_lines.append(line)

    # sort horizontal lines by y coordinate
    horizontal_lines.sort(key=lambda x: x[0][1])

    # draw rectangle in between the horizontal lines with offset of 10 pixels
    for index, line in enumerate(horizontal_lines):
        if index < len(horizontal_lines) - 1:
            shape = [(line[0][0], line[0][1] + 10), (line[1][0], horizontal_lines[index + 1][0][1] - 10)]
            draw.rectangle(shape, fill='white', outline='blue', width=2)

    for line in pred_lines:
        shape = [(line[0][0], line[0][1]), (line[1][0], line[1][1])]
        draw.line(shape, fill='red', width=2)

    img.save(png_path, 'PNG')
    

def image_enhance(img_gray, threshold = (50,100), clipLimit=2.0, tileGridSize=(8, 8)):
    lower_threshold, upper_threshold = threshold
    img_clipped = np.clip(img_gray, lower_threshold, upper_threshold)
    img_scaled = ((img_clipped - lower_threshold) / (upper_threshold - lower_threshold) * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit, tileGridSize)
    highlighted_img = clahe.apply(img_scaled)
    return highlighted_img

def process_lines(lines):
    # segment lines into horizontal and vertical lines
    horizontal_lines = []
    vertical_lines = []
    extend_lines = []

    for line in lines:
        angle = np.arctan2((line[1][1] - line[0][1]), (line[1][0] - line[0][0]))
        angle = angle * 180 / np.pi

        if -45 <= angle <= 45:
            horizontal_lines.append(line)
        else:
            vertical_lines.append(line)

    # sort horizontal lines by y coordinate
    horizontal_lines.sort(key=lambda x: x[0][1])

    # sort vertical lines by x coordinate
    vertical_lines.sort(key=lambda x: x[0][0])

    # longest horizontal line
    longest_horizontal_line = max(horizontal_lines, key=lambda x: calculate_length(x))
    x_start = longest_horizontal_line[0][0]
    x_end = longest_horizontal_line[1][0]

    # replace horizontal lines with longest horizontal line along the same y coordinate
    for i in range(len(horizontal_lines)):
        extend_lines.append(((x_start, horizontal_lines[i][0][1]), (x_end, horizontal_lines[i][0][1]))) 

    remaining_lines = extend_lines.copy()
    
    while len(remaining_lines) > 0:
        current_line = remaining_lines[0]
        remaining_lines.remove(current_line)

        for line in remaining_lines:
            dist = calculate_distance(current_line, line)
            # print(dist)
            if dist < 100:
                # get mid x coordinate of line
                mid_x = (line[0][0] + line[1][0]) / 2
                extend_lines.append(((mid_x, current_line[0][1]), (mid_x, line[0][1])))

    extend_lines = np.array([list(line) for line in extend_lines])

    return extend_lines


def draw_rectangles(img, lines,color = 'yellow', alpha=0.5):
    horizontal_lines = []
    for line in lines:
        angle = np.arctan2((line[1][1] - line[0][1]), (line[1][0] - line[0][0]))
        angle = angle * 180 / np.pi
        if -45 <= angle <= 45:
            horizontal_lines.append(line)
    horizontal_lines.sort(key=lambda x: x[0][1])
    
    fig, ax = plt.subplots(figsize=(img.shape[1] / 100, img.shape[0] / 100))
    ax.imshow(img, cmap='gray')

    # draw rectangles in between lines
    for i in range(len(horizontal_lines) - 1):
        x = [horizontal_lines[i][0][0] + 5, horizontal_lines[i][1][0] - 5, horizontal_lines[i+1][1][0] - 5, horizontal_lines[i+1][0][0] + 5]
        y = [horizontal_lines[i][0][1] + 5, horizontal_lines[i][1][1] + 5, horizontal_lines[i+1][1][1] - 5, horizontal_lines[i+1][0][1] - 5]
        
        ax.fill(x, y, color=color, alpha=alpha)

    # draw lines
    for line in lines:
        ax.plot((line[0][0], line[1][0]), (line[0][1], line[1][1]), color=color, alpha=1, linewidth=2)

    # save image
    plt.axis('off')
    plt.savefig('output/rectangles.png', bbox_inches='tight', pad_inches=0, dpi=100)
    plt.close()


if __name__ == "__main__":
    img_folder = 'output/'
    out_folder = 'output_img/'
    clipLimit=2.0
    tileGridSize=(8, 8)
    threshold = (50,100)
    device = 'cpu'
    merge_threshold=70
    orientation_threshold=3
    over_threshold=50
    extension_distance=90

    Parser = argparse.ArgumentParser()

    Parser.add_argument('--img_folder', default=img_folder, help='Path to input image folder')
    Parser.add_argument('--out_folder', default=out_folder, help='Path to output image folder')
    Parser.add_argument('--clipLimit', default=clipLimit, help='clipLimit for CLAHE')
    Parser.add_argument('--tileGridSize', default=tileGridSize, help='tileGridSize for CLAHE')
    Parser.add_argument('--threshold', default=threshold, help='threshold for CLAHE')
    Parser.add_argument('--device', default=device, help='device to run model on')
    Parser.add_argument('--merge_threshold', default=merge_threshold, help='threshold for merging lines')
    Parser.add_argument('--orientation_threshold', default=orientation_threshold, help='threshold for orientation')
    Parser.add_argument('--over_threshold', default=over_threshold, help='threshold for overlapping')
    Parser.add_argument('--extension_distance', default=extension_distance, help='distance for extending lines')

    Args = Parser.parse_args()
    img_folder = Args.img_folder
    out_folder = Args.out_folder
    clipLimit = Args.clipLimit
    tileGridSize = Args.tileGridSize
    threshold = Args.threshold
    device = Args.device
    merge_threshold = Args.merge_threshold
    orientation_threshold = Args.orientation_threshold
    over_threshold = Args.over_threshold
    extension_distance = Args.extension_distance

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    for img_name in os.listdir(img_folder):

        img_path = os.path.join(img_folder, img_name)

        # enhance image
        print("Enhancing image: ", img_name)
        img = cv2.imread(img_path)
        org_img = img.copy()
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_gray = image_enhance(img_gray,threshold=threshold, clipLimit=clipLimit, tileGridSize=tileGridSize)
        org_path = os.path.join(out_folder, img_name.split('.')[0] + '_1e.png')
        cv2.imwrite(org_path, img_gray)

        # Process the image
        print("Processing image: ", img_name)
        img, out, pred_lines = process_image_with_deeplsd(img_gray, out_folder,device=device )
        prc_path = os.path.join(out_folder, img_name.split('.')[0] + '_2p.png')
        # save_image_with_lines(img, pred_lines, prc_path)

        # Remove overlapping lines
        print("Removing overlapping lines")
        pred_lines = merge_lines_within_distance(pred_lines, merge_threshold=merge_threshold, orientation_threshold=orientation_threshold)
        pred_lines = remove_overlapping_lines(pred_lines, over_threshold=over_threshold)
        ovr_path = os.path.join(out_folder, img_name.split('.')[0] + '_3o.png')
        # save_image_with_lines(img, pred_lines, ovr_path)
        

        # Merge lines within a distance
        print("Merging lines within a distance")
        pred_lines = merge_lines_within_distance(pred_lines, merge_threshold=merge_threshold, orientation_threshold=orientation_threshold)
        
        mrg_path = os.path.join(out_folder, img_name.split('.')[0] + '_5m.png')
        # save_image_with_lines(img, pred_lines, mrg_path)

        # Extend the lines
        print("Extending the lines")
        pred_lines = extend_all_lines(pred_lines, extension_distance=extension_distance)
        ext_path = os.path.join(out_folder, img_name.split('.')[0] + '_6ex.png')
        # save_image_with_lines(img, pred_lines, ext_path)
        

        # Process the lines
        print("Processing the lines")
        pred_lines = process_lines(pred_lines)
        prc_path = os.path.join(out_folder, img_name.split('.')[0] + '_7p.png')
        # save_image_with_lines(img, pred_lines, prc_path)

        # Remove overlapping lines
        print("Removing overlapping lines")
        pred_lines = merge_lines_within_distance(pred_lines, merge_threshold=merge_threshold, orientation_threshold=orientation_threshold)
        pred_lines = remove_overlapping_lines(pred_lines, over_threshold=over_threshold)
        ovr_path = os.path.join(out_folder, img_name.split('.')[0] + '_8f.png')
        # save_image_with_lines(org_img, pred_lines, ovr_path)

        final_svg = os.path.join(out_folder, img_name.split('.')[0] + '_final.svg')
        save_svg(pred_lines, img, final_svg)

        final_png = os.path.join(out_folder, img_name.split('.')[0] + '_final.png')
        save_png(pred_lines, img, final_png)

        # Draw rectangles
        print("Drawing rectangles")
        draw_rectangles(org_img, pred_lines, color='yellow', alpha=0.3)
        