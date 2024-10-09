import numpy as np
import cv2
from cv2 import aruco

# Use an AprilTag dictionary
aruco_dict = aruco.Dictionary_get(aruco.DICT_APRILTAG_36h11)
Tag_dim = 4.00
Tag_spacing = 0.8

# Creating a grid board with one marker
grid = aruco.GridBoard_create(1, 1, Tag_dim, Tag_spacing, aruco_dict)
layout = grid.draw((250, 250))

# Camera matrix and distortion coefficients
K = np.array([[3.28189532e+03, 0.00000000e+00, 1.96178639e+03],
              [0.00000000e+00, 3.26287780e+03, 1.54649719e+03],
              [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

distortion_vals = np.array([[0.06594108, 0.09081337, 0.00118874, 0.00112097, -0.89241043]])

# Defining 3D points of a cube of size 4x4x4
Cordinates = np.float32([[-2, -2, 0], [-2, 2, 0], [2, 2, 0], [2, -2, 0],
                         [-2, -2, 4], [-2, 2, 4], [2, 2, 4], [2, -2, 4]])


def process_image(image_path, output_path):
    # Read and convert the image to grayscale
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Image not found or cannot be opened at '{image_path}'")
        return
    
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect markers
    corners, ids, rejected = aruco.detectMarkers(img_gray, aruco_dict, parameters=aruco.DetectorParameters_create())

    if corners:
        # Estimate pose of the marker
        rotation, translation, _ = aruco.estimatePoseSingleMarkers(corners[0], Tag_dim, K, distortion_vals)

        # Draw detected marker
        aruco.drawDetectedMarkers(image, corners)

        # Project 3D points to image plane
        pts, jac = cv2.projectPoints(Cordinates, rotation, translation, K, distortion_vals)
        pts = np.int32(pts).reshape(-1, 2)

        '''
        pts=[[1311 2095]
            [1258 1041]
            [2169  998]
            [2189 2302]
            [ 262 2434]
            [ 130 1039]
            [1043  964]
            [1151 2866]]
        '''
        # Draw the bottom square
        cv2.drawContours(image, [pts[:4]], -1, (0, 255, 0), 2)
        # Draw vertical lines
        for i, j in zip(range(4), range(4, 8)):
            cv2.line(image, tuple(pts[i]), tuple(pts[j]), (255, 0, 0), 3)
        # Draw the top square
        cv2.drawContours(image, [pts[4:]], -1, (0, 0, 255), 3)

    # Save the modified image
    cv2.imwrite(output_path, image)

    # Display the image in a window
    cv2.namedWindow("image", cv2.WINDOW_NORMAL)  # Create a resizable window
    cv2.resizeWindow("image", 400, 400)  # Resize window to 400x400 pixels
    cv2.imshow("image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Process Left and Right Perspective Images
process_image('/home/jithin/Desktop/Perception /HW1/Q4/TestImages/IMG_0190.jpg',
              '/home/jithin/Desktop/Perception /HW1/Q4/TestImages/Produced Image/Img_LP.jpg')

process_image('/home/jithin/Desktop/Perception /HW1/Q4/TestImages/IMG_0191.jpg',
              '/home/jithin/Desktop/Perception /HW1/Q4/TestImages/Produced Image/Img_RP.jpg')
