import numpy as np
import cv2
import random
import collections

############################### Task 1 ########################################
def undistort(img):
    size = img.shape[1::-1]
    camera_matrix = np.array( [[1.25626873e+03, 0, 6.91796054e+02], [0, 1.27958161e+03, 4.58611298e+02], [0, 0, 1]])
    dist_coeffs = np.array([[-2.69023741e-01, -2.47995232e-01, -1.21466540e-02,  7.29593971e-04, 7.54683988e-01]])
    alpha = 0.
    rect_camera_matrix = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, size, alpha)[0]
    map1, map2 = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, np.eye(3), rect_camera_matrix, size, cv2.CV_32FC1)
    undistorted_img = cv2.remap(img, map1, map2, cv2.INTER_LINEAR)
    return undistorted_img

def get_aruco_coordinates(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    aruco_dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    aruco_parameters =  cv2.aruco.DetectorParameters()
    aruco_detector = cv2.aruco.ArucoDetector(aruco_dictionary, aruco_parameters)
    corners, ids, _ = aruco_detector.detectMarkers(gray_img)
    coordinates = collections.OrderedDict(sorted((ids[i][0], corners[i][0]) for i in range(len(ids))))
    return coordinates

def get_aruco_common(img1, img2):
    coor1 = get_aruco_coordinates(img1)
    coor2 = get_aruco_coordinates(img2)
    common_markers = set(coor1.keys()).intersection(set(coor2.keys()))

    src = [(int(item[0]), int(item[1])) for key, value in coor1.items() if key in common_markers for item in value]
    dst = [(int(item[0]), int(item[1])) for key, value in coor2.items() if key in common_markers for item in value]

    return src, dst

# Sorts the list of images using middle of the first marker.  
def sort_images(imgs):
    all_coordinates = []
    prev = None
    for idx, img in enumerate(imgs):
        coors = get_aruco_coordinates(img)
        if prev is None:
            prev = set(coors.keys())
        else:
            prev = set(coors.keys()).intersection(prev)
        all_coordinates.append(coors)

    chosen_coordinate = list(prev)[0]
    sorted_coordinates = []

    for idx, coor in enumerate(all_coordinates):
        mean = np.mean([coor[chosen_coordinate][i][0] for i in range(4)])
        sorted_coordinates.append((mean, idx))

    sorted_coordinates.sort(reverse=True)
    sorted_indices = [imgs[idx] for _, idx in sorted_coordinates]

    return sorted_indices
###############################################################################

############################### Task 2 ########################################
# Projection with inverse homography and nearest neighbour implementation.
def apply_projective_transformation(img, transform_matrix, translate, name, wanted_shape=None):
    height, width = img.shape[:2] if not wanted_shape else wanted_shape[:2]

    inverse_transformation = np.linalg.inv(transform_matrix)
    ys, xs = np.mgrid[:height, :width]

    homogenous_vectors = np.stack([xs - translate[1], ys - translate[0], np.ones_like(xs)], axis=-1)
    source_vectors = (homogenous_vectors @ inverse_transformation.T)
    
    source_vectors /= source_vectors[..., -1, None]
    source_vectors = np.rint(source_vectors).astype(int)

    transformed_img = get_black_area(img, source_vectors)

    return transformed_img

# Add black area to not crop the image.
def get_black_area(img, vectors):
    result = np.zeros_like(vectors)
    for i, row in enumerate(vectors):
        for j, (x, y, _) in enumerate(row):
            if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
                result[i, j, :] = img[int(y), int(x), :]

    return result
###############################################################################

############################### Task 3 ########################################
def find_projective_transformation(src, dst):
    src = np.column_stack((src, np.ones(len(src))))
    dst = np.column_stack((dst, np.ones(len(dst))))

    A = []
    for i in range(len(src)):
        A.append([-src[i, 0], -src[i, 1], -1, 0, 0, 0, src[i, 0]*dst[i, 0], src[i, 1]*dst[i, 0], dst[i, 0]])
        A.append([0, 0, 0, -src[i, 0], -src[i, 1], -1, src[i, 0]*dst[i, 1], src[i, 1]*dst[i, 1], dst[i, 1]])

    A = np.array(A)
    _, _, V = np.linalg.svd(A)

    transformation = V[-1, :].reshape(3, 3)
    transformation /= transformation[2, 2]
    return transformation

# Task 3 tests
def test_find_projective_transformation(num_tests=20):
    for _ in range(num_tests):
        # Generate homography
        random_homography = np.random.rand(3, 3)
        random_homography[2, 2] = 1

        # And generate points
        num_points = 10
        src = np.random.rand(num_points, 2)
        src = np.column_stack((src, np.ones(num_points)))

        homogenized_dst = np.dot(random_homography, src.T).T
        dst = homogenized_dst[:, :2] / homogenized_dst[:, 2:]

        estimated_homography = find_projective_transformation(src[:, :2], dst)
        assert np.allclose(random_homography, estimated_homography, atol=0.1), "Test failed!"

    print("All tests for task 3 passed!")
###############################################################################

############################### Task 5 ########################################
def stitch_imgs(img2, img1, transformation, name):
    translate, new_shape = find_size_and_translate(
        img1.shape, 
        img2.shape, 
        transformation
    )
    new_shape += tuple([3])

    img1_resized = np.zeros(new_shape, np.uint8)
    # Move image to give space to the transformed image.
    img1_resized[translate[0]:translate[0]+img1.shape[0], translate[1]:translate[1]+img1.shape[1]] = img1

    img2_transformed = apply_projective_transformation(img2, transformation, translate, name, new_shape)

    stitched_img = blend_imgs(img1_resized, img2_transformed, translate, img1.shape)

    # Return image and translation coefs to join two parts of panorama.
    return stitched_img, translate

def is_black(img_coor):
    return img_coor[0] == img_coor[1] == img_coor[2] == 0

# Applies weighted average when stitching. 
def blend_imgs(img1, img2, translate, org_shape):
    h, w, _ = img1.shape
    blending_width = 40
    blending_height = 20
    
    mask = np.ones((h, w))
    # Make second img visible.
    mask[0:translate[0], :] = 0
    mask[translate[0]+org_shape[0]:-1, :] = 0
    mask[:, 0:translate[1]] = 0
    mask[:, translate[1]+org_shape[1]:-1] = 0
    
    gradient = np.linspace(0, 1, blending_width)
    for i in range(org_shape[0]):
        # Apply gradient from left to right.
        if not is_black(img2[translate[0]+i][translate[1]]):
            mask[translate[0]+i, translate[1]:translate[1]+blending_width] = gradient
        # And from right to left.
        if not is_black(img2[translate[0]+i][translate[1]+org_shape[1]-blending_width]):
            mask[translate[0]+i, translate[1]+org_shape[1]-blending_width:translate[1]+org_shape[1]] = gradient[::-1]


    gradient = np.linspace(0, 1, blending_height)
    for i in range(org_shape[1]):
        # Apply gradient from bottom to top.
        if translate[0]+blending_height < img2.shape[0] and not is_black(img2[translate[0]][translate[1]+i]):
            mask[translate[0]:translate[0]+blending_height, translate[1]+i] = np.transpose(gradient)
        
        # And from top to bottcom.
        if translate[0]+org_shape[0] < img2.shape[0] and not is_black(img2[translate[0]+org_shape[0]][translate[1]+i]):
            mask[translate[0]+org_shape[0]-blending_height:translate[0]+org_shape[0], translate[1]+i] = np.transpose(gradient[::-1])

    mask = np.expand_dims(mask, 2)

    # Apply mask.
    stitched_img = img1.copy() * mask + (1 - mask) * img2.copy()
    
    return stitched_img

def find_size_and_translate(base_shape, second_shape, transformation): 
    sh, sw, _ = second_shape 
    bh, bw, _ = base_shape

    # Transform the corners of the second shape
    transformed_corners = transformation @ np.vstack([np.array([[0, 0, sw, sw], [0, sh, 0, sh]]), np.ones((1, 4))])
    transformed_corners /= transformed_corners[-1, :]
    possible_corners = np.hstack([np.array([[0, 0, bw, bw], [0, bh, 0, bh]]), transformed_corners[:-1, :]])

    # Calculate translate and new size.
    left_top = np.min(possible_corners, axis=1)
    right_bottom = np.max(possible_corners, axis=1)
    translate_origin = tuple(np.rint(-left_top).astype(int))
    new_size = tuple(np.ceil(right_bottom - left_top).astype(int))

    return translate_origin[::-1], new_size[::-1]
###############################################################################

############################### Task 7 ########################################
import random
def ransac(src_points, dst_points, k=1000, inliers_error=15):
    best_transformation = None
    best_point_count = 5
    best_sum = 1e15

    for _ in range(k):
        random_index = random.sample(list(range(len(src_points)//4 - 1)), 2)
        random_src = np.array(src_points[4*random_index[0] : 4*random_index[0] + 4] + src_points[4*random_index[1] : 4*random_index[1] + 4])
        random_dst = np.array(dst_points[4*random_index[0] : 4*random_index[0] + 4] + dst_points[4*random_index[1] : 4*random_index[1] + 4])

        transformation = find_projective_transformation(random_src, random_dst)
        if np.linalg.matrix_rank(transformation) < 3:
            continue

        # Apply the transformation to all source points.
        transformed_src = np.dot(np.column_stack((src_points, np.ones(len(src_points)))), transformation.T)
        transformed_src = transformed_src[:, :2] / transformed_src[:, 2, None]

        # Calculate distance.
        distances = np.linalg.norm(transformed_src - dst_points, axis=1)

        # Count inliers and sum.
        inliers = distances < inliers_error
        if (len(inliers) > best_point_count) or (len(inliers) == best_point_count and best_sum > np.sum(distances)):
            # Find the transformation based on all inliers.
            best_transformation = transformation #find_projective_transformation(np.array(src_points)[inliers], np.array(dst_points)[inliers])
            best_point_count = len(inliers) 
            best_sum = np.sum(distances)

    return best_transformation
###############################################################################

############################### Task 4/5/6 ####################################
def manual_stitching(path):
    print("Task 4")
    img1 = cv2.imread(path + "hw11.jpeg")
    img2 = cv2.imread(path + "hw12.jpeg")
    src_manual = [(623, 528), (617, 361), (1341, 629), (1277, 246)]
    dst_manual = [(133, 545), (177, 380), (879, 671), (884, 256)]
    transform_matrix = find_projective_transformation(src_manual, dst_manual)
    print("Task 5")
    stitch_imgs(img1, img2, transform_matrix, "task_5")
    print("Task 6")
    src, dst = get_aruco_common(img1, img2)
    transform_matrix = find_projective_transformation(src, dst)
    stitch_imgs(img1, img2, transform_matrix, "task_6")
###############################################################################

############################### Task 7 ########################################
def joint_two_panorama_parts(img, translate, name):
    print("Join panorama " + name)
    h1, w1, _ = img[0].shape 
    h2, w2, _ = img[1].shape
    s = [[h1, h2],[w1, w2]]
    new_shape = [max(h1, h2), max(w1, w2), 3]
    move = np.array([[0, 0], [0, 0]])
    
    # Find shape and how to move images.
    for i in range(2):
        diff = abs(translate[0][i] - translate[1][i])
        if  translate[0][i] >= translate[1][i]:
            move[1][i] = diff
            new_shape[i] = max(s[i][1] + diff, new_shape[i])
        else:
            move[0][i] = diff
            new_shape[i] = max(s[i][0] + diff, new_shape[i])

    new_shape = tuple(new_shape)
    final = np.zeros(new_shape, np.uint8)

    final[move[0][0] : move[0][0] + s[0][0], move[0][1] : move[0][1] + s[1][0]] = img[0]

    for i in range(s[0][1]):
        ti = move[1][0] + i
        for j in range(s[1][1]):
            tj = move[1][1] + j
            if is_black(final[ti][tj]):
                final[ti][tj] = img[1][i][j]
            elif not is_black(img[1][i][j]):
                final[ti][tj] = (final[ti][tj] + img[1][i][j]) / 2

    cv2.imwrite('./' + name +'.jpg', final)

def stitch_for_subset(path, subset, final_name):
    imgs = []
    for nr in subset:
        img = cv2.imread(path + "hw" + nr +".jpeg")
        undistorted = undistort(img)
        imgs.append(undistorted)

    stitches = []
    for i, j in [(0, 1), (2, 1)]:
        src, dst = get_aruco_common(imgs[i], imgs[j])
        best_matrix = ransac(src, dst)
        stitch_and_translate  = stitch_imgs(imgs[i], imgs[j], best_matrix, str(subset[i]) + str(subset[j]))
        stitches.append(stitch_and_translate)
    
    joint_two_panorama_parts([stitches[0][0], stitches[1][0]], [stitches[0][1], stitches[1][1]], final_name)

def stitch_subsets(path):
    subset1 = ['11', '12', '13']
    stitch_for_subset(path, subset1, 'task_7_1')
    subset2 = ['11', '14', '13']
    stitch_for_subset(path, subset2, 'task_7_2')
###############################################################################


if __name__ == "__main__":
    test_find_projective_transformation()
    path = "imgs/"
    manual_stitching(path)
    stitch_subsets(path)