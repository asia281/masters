"""
Stub for homework 2
"""
import time
import random
import numpy as np
import mujoco
from mujoco import viewer
import cv2

image_center = 320

# Ranges in hsv colors for blue, red and green
BLUE_RANGE = [np.array([110, 50, 50]), np.array([130, 255, 255])]
RED_RANGE = [np.array([0, 100, 50]), np.array([10, 255, 255])]
GREEN_RANGE = [np.array([36, 25, 25]), np.array([86, 255,255])]

model = mujoco.MjModel.from_xml_path("car.xml")
renderer = mujoco.Renderer(model, height=480, width=640)
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)
viewer = viewer.launch_passive(model, data)

def sim_step(forward, turn, steps=1000, view=True):
    data.actuator("forward").ctrl = forward
    data.actuator("turn").ctrl = turn
    for _ in range(steps):
        step_start = time.time()
        mujoco.mj_step(model, data)
        if view:
            viewer.sync()
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step / 10)
    renderer.update_scene(data, camera="camera1")
    image = renderer.render()
    return image

def task_1_step(turn):
    image = sim_step(0.1, turn, steps=200, view=False)
    return image

def crop_image(image):
    return image[0:400, :]

# Find occurences of a given range (which corresponds to a color)
def add_mask(image, lower_margin, upper_margin):
    return cv2.inRange(image, lower_margin, upper_margin)

def find_color_occurences(image, color_range):
    hsv_image = cv2.cvtColor(crop_image(image), cv2.COLOR_RGB2HSV)
    mask = add_mask(hsv_image, color_range[0], color_range[1])
    return mask

# Find the contours and keep the largest one
def find_marker(image, mask):
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return None

    c = max(contours, key=cv2.contourArea)
    return cv2.minEnclosingCircle(c)

def find_marker_ball(image):
    mask = find_color_occurences(image, RED_RANGE)
    return find_marker(image, mask)

def ball_visible(image, threshold = 20, distance = 0.2):
    markers = find_marker_ball(image)
    return markers is not None and (image_center -  threshold - 40 < markers[0][0] < image_center - threshold)

def while_ball_invisible(image, turn, threshold):
    while not ball_visible(image, threshold):
        image =  task_1_step(turn)
    return image

def localize_ball(image, turn, threshold = 20) :
    image = while_ball_invisible(image,  turn, threshold)
    return image, find_marker_ball(image)

def task_1():
    step_size = 0.1
    steps = random.randint(0, 2000)
    image = sim_step(0, 0.1, steps, view=False)

    # TODO: change the lines below,
    # for car control, you should use task_1_step(turn) function
    # you can change anything below this line
    org_ball_pos = data.body("target-ball").xpos

    image, marker = localize_ball(image, -0.1)
    while marker[1] < 55:
        for i in range(100):
            if  marker is None or marker[0][0] < 150:
                image, marker = localize_ball(image, 0.1)
            if marker[1] > 55:
                break
            image = task_1_step(0)
            marker = find_marker_ball(image)
        if find_marker_ball(image)[1] > 35:
            image, marker = localize_ball(image, -0.1, 60)
        else:
            image, marker = localize_ball(image, -0.1)

    print("final distance:", np.linalg.norm(data.body("car").xpos - data.body("target-ball").xpos))
    # Check distance and it car touched ball.
    assert np.linalg.norm(data.body("car").xpos - data.body("target-ball").xpos) < 0.2 and (data.body("target-ball").xpos.all() == org_ball_pos.all())
    # at the end, your car should be close to the red ball (0.2 distance is fine)
    # data.body("car").xpos) is the position of the car

##################### Task 2 ###############################

def green_is_visible(image, threshold=30, shift = 100):
    image = crop_image(image)
    mask = find_color_occurences(image, GREEN_RANGE)
    markers = find_marker(image, mask)
    return  markers is not None and image_center - threshold - shift < markers[0][0]  < image_center + threshold - shift, markers

def find_green(image):
    return green_is_visible(image, threshold = 10, shift = 0)

def ball_visible_2(image, threshold = 20, margin = 20):
    markers = find_marker_ball(image)
    return markers is not None and image_center -  threshold - margin < markers[0][0] < image_center - threshold

def ball_invisible_2(image,  threshold=0):
    while not ball_visible_2(image, threshold):
        markers = find_marker_ball(image)
        if markers is None or markers[0][0] < 250:
            image = sim_step(0,  0.04)
        else:
            image = sim_step(0,  -0.003)
    return image

def task_2():
    sim_step(0.5, 0, 1000, view=True)
    speed = random.uniform(0.3, 0.5)
    turn = random.uniform(-0.2, 0.2)
    image = sim_step(speed, turn, 1000, view=True)
    # TODO: change the lines below,
    # you should use sim_step(forward, turn) function
    # you can change the speed and turn as you want
    # do not change the number of steps (1000)

    deg_90 = 0.12
    image = ball_invisible_2(image)
    markers = find_marker_ball(image)
    steps = []
    while markers[1] < 30:
        steps.append(2 / markers[1])
        image = sim_step(2 / markers[1], 0)
        image = ball_invisible_2(image)
        markers = find_marker_ball(image)

    ball_width = 0.8
    sim_step(0, deg_90)
    sim_step(ball_width /2, 0)
    sim_step(0, -deg_90)
    sim_step(ball_width, 0)
    sim_step(0, -deg_90)
    sim_step(ball_width /2 + 0.1, 0)
    image = sim_step(0, -deg_90)
    markers = find_marker_ball(image)

    while markers[1] < 50:
        steps.append(2 / markers[1])
        image = sim_step(2 / markers[1], 0)
        image = ball_invisible_2(image)
        markers = find_marker_ball(image)

    while markers[0][0] - 40 < image_center:
        image = sim_step(0, 0.01)
        markers = find_marker_ball(image)

    for _ in range(10):
        image = sim_step(0.05,  0)

    visible, markers = find_green(image)
    print(markers)

    while markers is None or markers[0][0] > 475:
        image = sim_step(0.01, 0.01)
        image = sim_step(0.01, -0.0105)
        visible, markers = find_green(image)

    while markers[1] < 50:
        if markers[0][0] > image_center:
            image = sim_step(0.02, -0.03)
            image = sim_step(0.02,  0.05)
        else:
            image = sim_step(0.02, 0.03)
            image = sim_step(0.02, - 0.05)
        # image = sim_step(1.07,- 0.025)
        # image = sim_step(0.07, -0.013)
        # image = sim_step(0.05, 0.01)
        visible, markers = find_green(image)
    # at the end, red ball should be close to the green box (0.25 distance is fine)

##################### Task 3 ###############################
drift = 0

def task3_step(forward, turn, steps=200, view=False):
    return sim_step(forward, turn + drift, steps=steps, view=view)

def my_step(forward, turn, my_drift, steps=200):
    return  crop_image(task3_step(forward, turn - my_drift, steps=steps))

def blue_is_visible(image, threshold=30):
    image = crop_image(image)
    mask = find_color_occurences(image, BLUE_RANGE)
    markers = find_marker(image, mask)
    return  (markers is not None and image_center - threshold + 60 < markers[0][0]  < image_center + threshold + 60), markers

def find_drift():
    image = task3_step(0, 0)
    image = while_ball_invisible(image, 0.11, -30)
    l, r = -0.1, 0.1
    mid = 0
    while l + 0.005 < r:
        prev_marker = find_marker_ball(image)[0][0]
        image = task3_step(0, mid)
        curr_marker = find_marker_ball(image)[0][0]
        if prev_marker < curr_marker:
            r = mid
        else:
            l = mid
        mid = (l + r) / 2
        image = while_ball_invisible(image, 0.105, -30)

    print(drift, mid)
    return -mid, image

def go_to_green(my_drift, image):
    visible, markers = green_is_visible(image, 5)
    while not visible:
        if markers == None:
            image = my_step(0, 0.1, my_drift)
        else:
            image = my_step(0, 0.01, my_drift)
        visible, markers = green_is_visible(image, 5)
    while markers[1] < 100:
        image = my_step(0.2, 0, my_drift)
        visible, markers = green_is_visible(image, 5)

    image = my_step(0.8, 0, my_drift)
    image = my_step(0.6, 0, my_drift)
    return image

def go_to_blue(my_drift, image, blue_step_size):
    visible, markers = blue_is_visible(image, 5)
    while not visible:
        image = my_step(0, 0.01, my_drift)
        visible, markers = blue_is_visible(image, 5)
    cnt = 0
    while markers[1] < 110:
        cnt += 1
        image = my_step(blue_step_size, 0, my_drift)
        visible, markers = blue_is_visible(image, 2)

    return cnt

def task_3():
    global drift
    drift = np.random.uniform(-0.1, 0.1)
    # TODO: change the lines below,
    # you should use task3_step(forward, turn, steps) function
    my_drift, image = find_drift()

    image = go_to_green(my_drift, image)

    blue_step_size = 0.2

    cnt = go_to_blue(my_drift, image, blue_step_size)
    my_step(0.27 - cnt * blue_step_size / 2, 0, my_drift)

    box1_pos, box2_pos = data.body("target-box-1").xpos, data.body("target-box-2").xpos
    car_pos = data.body("car").xpos
    print("final postions of the car and boxes :", car_pos, box1_pos, box2_pos)
    assert box1_pos[0] > car_pos[0] > box2_pos[0] and box1_pos[1] > car_pos[1] > box2_pos[1]
    # at the end, car should be between the two boxes
