import cv2
import numpy as np
import glob

pattern_size = (8, 5)
checker_width = 30 

def find_chessboard_corners(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

    if ret:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        cv2.drawChessboardCorners(img, pattern_size, corners, ret)
        cv2.imshow('Chessboard Corners', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print(f"No chessboard found in {image_path}")

def calibrate_camera(images_folder):
    objpoints = [] 
    imgpoints = []

    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) * checker_width

    for image_path in glob.glob(images_folder + '/*.jpg'):
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)

    ret, camera_matrix, dist_coeffs, _, _ = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    from decimal import Decimal
    
    return camera_matrix, dist_coeffs

def undistort_images(images_folder, camera_matrix, dist_coeffs):
    for image_path in glob.glob(images_folder + '/*.jpg'):
        img = cv2.imread(image_path)

        h, w = img.shape[:2]
        new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
        undistorted_img = cv2.undistort(img, camera_matrix, dist_coeffs, None, new_camera_matrix)

        side_by_side = np.concatenate((img, undistorted_img), axis=1)
        cv2.imshow('Original vs Undistorted', side_by_side)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    images_folder = 'data'
    float_formatter = "{:.10f}".format
    np.set_printoptions(formatter={'float_kind': float_formatter})

    for image_path in glob.glob(images_folder + '/*.jpg'):
        print(image_path)
        find_chessboard_corners(image_path)

    camera_matrix, dist_coeffs = calibrate_camera(images_folder)
    print("Camera Matrix:\n", camera_matrix)
    print("Distortion Coefficients:\n", dist_coeffs)

    undistort_images(images_folder, camera_matrix, dist_coeffs)

