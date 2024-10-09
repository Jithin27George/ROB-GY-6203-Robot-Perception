import numpy as np
import cv2
from cv2 import aruco
import os

# Function to process images and detect corners and ids for AprilTags
def process_images(img_dir, aruco_dict, arucoParams):
    corners_list = []
    id_list = []
    counter = []
    
    # Loop over images in the directory
    for img_name in os.listdir(img_dir):
        img_path = os.path.join(img_dir, img_name)
        image = cv2.imread(img_path)
        if image is None:
            print(f"Failed to read image {img_name}")
            continue
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = aruco.detectMarkers(img_gray, aruco_dict, parameters=arucoParams)
        
        if ids is not None:
            # Accumulate detected corners and IDs
            corners_list.extend(corners)
            id_list.extend(ids.flatten())
            counter.append(len(ids))
        else:
            # No markers detected in this image
            print(f"No markers detected in image {img_name}")
    
    # Prepare the ID list as a NumPy array of shape (N, 1)
    id_list = np.array(id_list).reshape((-1, 1))
    counter = np.array(counter)
    
    return corners_list, id_list, counter, img_gray.shape[::-1]

# Function to perform camera calibration using AprilTags
def calibrate_camera(corners_list, id_list, counter, grid, img_shape):
    ret, K, distortion_vals, rvecs, tvecs = aruco.calibrateCameraAruco(
        corners_list,
        id_list,
        counter,
        grid,
        img_shape,
        None,
        None
    )
    return K, distortion_vals

# Use an AprilTag dictionary
aruco_dict = aruco.Dictionary_get(aruco.DICT_APRILTAG_36h11)

# Dimensions in centimeters
Tag_dim = 2.00
Tag_spacing = 0.8

# Create detector parameters for AprilTags
arucoParams = aruco.DetectorParameters_create()

# Create a grid board with AprilTags
grid = aruco.GridBoard_create(7, 5, Tag_dim , Tag_spacing, aruco_dict)
layout = grid.draw((4000, 4000))
print(layout.shape)
# Save the board image
cv2.imwrite('/home/jithin/Desktop/AprilTag/collage_0_34.jpg', layout)

# Directories of images for different calibration distances
img_dirs = ['/home/jithin/Desktop/AprilTag/calibimg_1mtr', '/home/jithin/Desktop/AprilTag/calibimg_1_2mtr']

for i, img_dir in enumerate(img_dirs):
    print(f"\nProcessing images from: {img_dir}")
    
    # Detect corners and IDs for AprilTags in the images
    corners_list, id_list, counter, img_shape = process_images(img_dir, aruco_dict, arucoParams)
    
    # Perform camera calibration
    K, distortion_vals = calibrate_camera(corners_list, id_list, counter, grid, img_shape)
    
    # Output the camera matrix and distortion coefficients
    print(f'\nCase {i+1}: Camera matrix (K):\n', K)
    print(f'Case {i+1}: Distortion coefficients (d):', distortion_vals.ravel())

