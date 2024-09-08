from itertools import count
import numpy as np
import cv2
import datetime
import argparse

process_var = 1  # Process noise variance
measurement_var = 1e4  # Measurement noise variance


class KalmanFilter:
    def __init__(self, process_var, measurement_var_x, measurement_var_y):
        # process_var: process variance, represents uncertainty in the model
        # measurement_var: measurement variance, represents measurement noise

        # Measurement Matrix
        self.H = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]])

        # Process Covariance Matrix
        self.Q = np.eye(6) * process_var

        # Measurement Covariance Matrix
        self.R = np.array(
            [
                [measurement_var_x, 0],
                [0, measurement_var_y],
            ]
        )

        # Initial State Covariance Matrix
        self.P = np.eye(6)

        # Initial State
        self.x = np.zeros(6)

    def predict(self, dt):
        # State Transition Matrix -- x, y, vx, vy, ax (const), ay (const)
        A = np.array([[1, 0, dt,  0, 0, 0], 
                      [0, 1, 0, dt, 0, 0], 
                      [0, 0, 1, 0, dt, 0], 
                      [0, 0, 0, 1, 0, dt], 
                      [0, 0, 0, 0, 1, 0], 
                      [0, 0, 0, 0, 0, 1]])
        # Predict the next state
        self.x = A @ self.x
        self.P = A @ self.P @ A.T + self.Q
        print(f"Predicted State: {self.x}")

    def update(self, measurement):
        # Update the state with the new measurement
        print(f"Measurement: {measurement}")
        y = measurement - self.H @ self.x
        print(f"y: {y}")
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x += K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P


def draw_uncertainty(kf, img):
    x, y = kf.x[:2]
    confidence_factor = 1.96  # For 90% confidence assuming a normal distribution (so 95% * 95%)

    std_x = np.sqrt(kf.P[0, 0])
    std_y = np.sqrt(kf.P[1, 1])

    half_length_x = confidence_factor * std_x
    half_length_y = confidence_factor * std_y

    top_left = (int(x - half_length_x), int(y - half_length_y))
    bottom_right = (int(x + half_length_x), int(y + half_length_y))

    cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 1)


class ClickReader:
    def __init__(self, process_var, measurement_var, window_name="Click Window"):
        self.window_name = window_name
        self.cur_time = datetime.datetime.now()
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        self.kf = KalmanFilter(process_var, measurement_var, measurement_var)

        self.img = 255 * np.ones((500, 500, 3), np.uint8)

    def mouse_callback(self, event, x, y, flags, param):
        # Check if the event is a left button click
        if event == cv2.EVENT_LBUTTONDOWN:
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"Time: {current_time}, Position: ({x}, {y})")
            new_time = datetime.datetime.now()
            self.kf.predict((new_time - self.cur_time).total_seconds())
            self.cur_time = new_time

            cv2.circle(self.img, (x, y), 2, (0, 0, 255), -1)  # Red color, filled circle
            self.kf.update(np.array((x, y)))
            print(f"Updated State: {self.kf.x}")

    def run(self):
        # Main loop to display the window
        while True:
            new_time = datetime.datetime.now()
            self.kf.predict((new_time - self.cur_time).total_seconds())
            self.cur_time = new_time

            cv2.circle(
                self.img, (int(self.kf.x[0]), int(self.kf.x[1])), 2, (255, 0, 0), -1
            )  # Blue color, filled circle

            img_draw = self.img.copy()

            print(img_draw.shape)

            draw_uncertainty(self.kf, img_draw)

            cv2.imshow(self.window_name, img_draw)

            # Exit on pressing the 'ESC' key
            if cv2.waitKey(1) & 0xFF == 27:
                break

        cv2.destroyAllWindows()


class PredefinedClickReader:
    def __init__(
        self,
        process_var,
        measurement_var_x,
        measurement_var_y,
        window_name="Click Window",
    ):
        self.window_name = window_name
        self.cur_time = datetime.datetime.now()
        cv2.namedWindow(self.window_name)
        self.kf = KalmanFilter(process_var, measurement_var_x, measurement_var_y)

        self.img = 255 * np.ones((500, 500, 3), np.uint8)

    def run(self, observation_generator):
        for dt, observation in observation_generator:
            self.kf.predict(dt)
            print(observation)
            if observation is not None:
                self.kf.update(observation)
                cv2.circle(
                    self.img,
                    (int(observation[0]), int(observation[1])),
                    2,
                    (0, 0, 255),
                    -1,
                )
            cv2.circle(
                self.img, (int(self.kf.x[0]), int(self.kf.x[1])), 2, (255, 0, 0), -1
            )
            img_draw = self.img.copy()
            draw_uncertainty(self.kf, img_draw)
            cv2.imshow(self.window_name, img_draw)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def parabola_generator():
    for x in range(0, 500, 1):
        if np.random.rand(1)[0] > 0.5:
            yield 1, None
        else:
            yield 1, np.array(
                [
                    x + np.random.randn(1)[0] * np.sqrt(1e2),
                    x * (500 - x) / 250 + np.random.randn(1)[0] * np.sqrt(4e2),
                ]
            )


class VideoReader:
    def __init__(
        self,
        process_var,
        measurement_var,
        video_path,
        window_name="Video Window",
        fps=29.97,
    ):
        self.video = cv2.VideoCapture(video_path)
        self.kf = KalmanFilter(process_var, measurement_var, measurement_var)
        self.window_name = window_name
        cv2.namedWindow(window_name)
        self.fps = fps

    # Finds red ball in the frame.
    def find_position(self, frame, prev=None):
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        value_normalized = hsv_frame[:, :, 2] / 255.0

        print(np.mean(value_normalized))

        if prev is not None and (prev - np.mean(value_normalized)) > 0.02:
            return None, prev


        lower_red = np.array([0, 70, 50])
        upper_red = np.array([10, 255, 255])
        mask1 = cv2.inRange(hsv_frame, lower_red, upper_red)
        lower_red = np.array([170, 70, 50])
        upper_red = np.array([180, 255, 255])
        mask2 = cv2.inRange(hsv_frame, lower_red, upper_red)
        mask = mask1 + mask2
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return None, np.mean(value_normalized)

        cnt = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(cnt)
        print("bounds", x, y, w, h)
        measurement = [x + w/2, y + h/2]

        return measurement, np.mean(value_normalized)

    def run(self):
        thr = 1e3
        self.kf.P[2, 2] = thr
        self.kf.P[3, 3] = thr
        self.kf.P[4, 4] = thr
        self.kf.P[5, 5] = thr

        ret, frame = self.video.read()
        if not ret:
            return

        initial_position, prev = self.find_position(frame)
        if initial_position is not None:
            self.kf.x[:2] = initial_position

        while True:
            ret, frame = self.video.read()
            if not ret:
                break

            observed_position, prev = self.find_position(frame, prev)
            if observed_position is not None:
                self.kf.update(observed_position)
                cv2.circle(frame, (int(observed_position[0]), int(observed_position[1])), 10, (0, 255, 0), -1)

            # dt is ~1/fps, it work well at line.mp4. To improve performance on sinewave.mp4, you can change it to 0.2.
            self.kf.predict(0.04)
            # Comment the if below to draw predicted positions for all timesteps
            if observed_position is None:
                predicted_position = self.kf.x[:2]
                cv2.circle(frame, (int(predicted_position[0]), int(predicted_position[1])), 10, (255, 255, 0), -1)

            cv2.imshow(self.window_name, frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        cv2.destroyAllWindows()


if __name__ == "__main__":
    # add an argument to decide between click, predefined and video
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default="click",
        choices=["click", "predefined", "video"],
        help="Mode to run the program in. Options are click, predefined, video",
    )
    args = parser.parse_args()
    if args.mode == "click":
        click_reader = ClickReader(process_var, measurement_var)
        click_reader.run()
    elif args.mode == "predefined":
        process_var = 1e-3
        # you can decrese these values when decresing dt in parabola_generator (to look more like on the photo) 
        # these values are taken from parabola generator 
        measurement_var_x = 1e2
        measurement_var_y = 4e2

        predefinedclicker = PredefinedClickReader(
            process_var, measurement_var_x, measurement_var_y
        )
        predefinedclicker.run(parabola_generator())
    else:
        assert args.mode == "video"
        video_reader = VideoReader(10, 10, "line.mp4")
        video_reader.run()
