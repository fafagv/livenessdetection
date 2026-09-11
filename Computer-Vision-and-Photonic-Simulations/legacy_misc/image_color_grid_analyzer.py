# --- LEGACY/MISC FILE ---
# This script does not belong to any module in the target repository
# architecture (Computer Vision / Photonic Simulations / Deep Learning).
# It is kept here unlinked, for reference only, and is not imported by
# anything in src/ or apps/. See legacy_misc/README.md for details.
# ------------------------

import cv2
import numpy as np


# Load an image from file
def load_image(image_path):
    # Load the image in BGR format
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not open or find the image.")
        return None
    return image


# Apply median filter to remove salt-and-pepper noise
def remove_noise(image):
    # Apply a median filter to reduce salt-and-pepper noise
    denoised_image = cv2.medianBlur(image, 5)
    return denoised_image


# Function to detect and return color pattern as matrix-like array
def detect_colors(image, grid_size=(4, 4)):
    # Convert the image from BGR to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define color ranges for common colors (adjust based on your images)
    color_ranges = {
        'red': [(0, 100, 100), (10, 255, 255)],
        'green': [(40, 50, 50), (80, 255, 255)],
        'blue': [(100, 100, 100), (130, 255, 255)],
        'yellow': [(20, 100, 100), (30, 255, 255)],
        'black': [(0, 0, 0), (180, 255, 30)],
        'white': [(0, 0, 200), (180, 30, 255)]
    }

    # Create a blank color matrix to store the results
    color_matrix = [['' for _ in range(grid_size[1])] for _ in range(grid_size[0])]

    # Split the image into grid cells
    h, w, _ = image.shape
    cell_height = h // grid_size[0]
    cell_width = w // grid_size[1]

    # Loop through each grid cell to detect the predominant color
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            # Extract the region of interest (cell)
            cell = hsv_image[i * cell_height:(i + 1) * cell_height, j * cell_width:(j + 1) * cell_width]

            # Detect predominant color by checking which color range the majority of pixels fall into
            for color, (lower, upper) in color_ranges.items():
                mask = cv2.inRange(cell, np.array(lower), np.array(upper))
                if np.sum(mask) > mask.size // 2:  # If more than half the pixels match this color
                    color_matrix[i][j] = color
                    break
            else:
                color_matrix[i][j] = 'unknown'

    return color_matrix


# Display the image with the grid overlay
def display_image_with_grid(image, grid_size=(4, 4)):
    h, w, _ = image.shape
    cell_height = h // grid_size[0]
    cell_width = w // grid_size[1]

    def display_image_with_grid(image, grid_size=(4, 4)):
        h, w, _ = image.shape
        cell_height = h // grid_size[0]
        cell_width = w // grid_size[1]

        # Draw grid lines on the image
        for i in range(1, grid_size[0]):
            cv2.line(image, (0, i * cell_height), (w, i * cell_height), (0, 255, 0),
                     2)  # Change color to green for visibility
        for j in range(1, grid_size[1]):
            cv2.line(image, (j * cell_width, 0), (j * cell_width, h), (0, 255, 0),
                     2)  # Change color to green for visibility

        cv2.imshow("Image with Grid", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# Main function to process the image
def process_image(image_path):
    # Load the image
    image = load_image(image_path)
    if image is None:
        return

    # Remove salt-and-pepper noise
    denoised_image = remove_noise(image)

    # Detect colors in the image
    color_matrix = detect_colors(denoised_image)

    # Display the original image with grid overlay
    display_image_with_grid(denoised_image)

    # Print the color matrix
    print("Detected color pattern:")
    for row in color_matrix:
        print(row)


# Test the function with a sample image
image_path = r"C:\Users\Asus\Documents\MATLAB\images\org_3.png"
process_image(image_path)
