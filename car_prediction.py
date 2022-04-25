from ast import Break
import cv2
import numpy as np
from py import process
from kalmanfilter import KalmanFilter
from car_detector import CarDetector

cap = cv2.VideoCapture("cut1.mp4")

# Load detector
car = CarDetector()

# Load Kalman filter to predict the trajectory
kf = KalmanFilter()


def normalized_cross_correlation(ch1, ch2):
    """
    Calculates similarity between 2 color channels
    Input:  ch1: channel 1 matrix
            ch2: channel 2 matrix
    Output: normalized cross correlation (scalar)
    """

    ch1_copy = (ch1) / np.linalg.norm(ch1)
    ch2_copy = (ch2) / np.linalg.norm(ch2)

    return np.sum(ch1_copy * ch2_copy)

# _____________________________________________________

# find best bb in new frame that matches old frame
# def find_best_bb(old_frame, new_frame, old_upL = (x, y), old_lowR = (x1, y1)):

def find_best_bb(old_frame, new_frame, x, y, x1, y1, gamma = 0.03):

    ref_img = old_frame[y:y1, x:x1]

    H = old_frame.shape[0]
    W = old_frame.shape[1]

    H_bb = y1 - y
    W_bb = x1 - x

    # find search bounds within 10% of old bb
    low_Xi = max(0, int(x * (1 - gamma)))
    high_Xi = min(W - 1, int(x * (1 + gamma)))

    low_Yj = max(0, int(y * (1 - gamma)))
    high_Yj = min(H - 1, int(y * (1 + gamma)))

    # do quadruple for loop, checking area is within 10% of previous, and constraint that u > i, and v > j
    # i, j are upper left, u, v are lower right

    best_correlation = 0
    best_offset = (0, 0, 0, 0)

    # print("X range:", low_Xi, high_Xi)
    # print("Y range:", low_Yj, high_Yj)

    for i in range(low_Xi, high_Xi):
        for j in range(low_Yj, high_Yj):
            # resize image to make same with ref, old frame
            dim = (ref_img.shape[1], ref_img.shape[0])

            try:
                temp_ref = cv2.resize(new_frame[j:y1, i:x1], dim, interpolation = cv2.INTER_AREA)
            except:
                print("error resizing")
            else:

                # compute correlation
                corr = normalized_cross_correlation(temp_ref, ref_img)

                # update best correlation and bounding box
                if (corr > best_correlation):
                    best_correlation = corr

                    x_diff = abs(i - x)
                    y_diff = abs(j - y)

                    best_offset = (i + x_diff * 3, j + y_diff * 3, x1 + x_diff * 3, y1 - y_diff * 3)
                        

    return best_offset    


if __name__ == "__main__":
    print("Processing video...")

    iter = 0
    x, y, x1, y1 = 330, 700, 810, 1050

    width = abs(x1 - x)
    height = abs(y1 - y)

    processed_frames = []
    coordinates = []

    while True:
        ret, frame = cap.read()
        if ret is False:
            print(f"Completed processing {iter} frames.")
            break

        if (iter > 0):
            x, y, x1, y1 = find_best_bb(old_frame, frame, x, y, x1, y1)              

            cx = (x + x1) / 2
            cy = (y + y1) / 2

            predicted = kf.predict(cx, cy)
            print("PREDICTED", predicted)
            print("Cs", cx, cy)
            print("COORDINATES", x, y, x1, y1)

            px = abs(predicted[0] - width / 2)
            px1 = abs(predicted[0] + width / 2)
            py = abs(predicted[1] - height / 2)
            py1 = abs(predicted[1] + height / 2)

            predicted = np.array([px, py, px1, py1])
            coordinates.append(predicted)
            print("PREDICTED", predicted[0], predicted[1], predicted[2], predicted[3])


        if iter % 1 == 0:
            print(f"frame {iter}")

        old_frame = frame
        processed_frames.append(old_frame)
        
        # cv2.rectangle(frame, (x, y), (x1, y1), (255, 0, 0), 4)

        # cv2.imshow("Frame", frame)
        # key = cv2.waitKey(3000)

        # if key == 27:
        #     break
        
        iter += 1
    
    processed_frames = np.array(processed_frames)
    coordinates = np.array(coordinates)
    
    print(coordinates.shape)

    cv2.imshow("Frame", processed_frames[0])
    cv2.waitKey(1)
    for i in range(1, processed_frames.shape[0]):
        cv2.rectangle(frame, (int(coordinates[i-1, 0]), int(coordinates[i-1, 1])), (int(coordinates[i-1, 2]), int(coordinates[i-1, 3])), (255, 0, 0), 4)
        cv2.imshow("Frame", processed_frames[i])


    while(True):
        choice = input("Enter r to replay or any key to quit")

        if choice == "r":
            cv2.imshow("Frame", processed_frames[0])
            for i in range(1, processed_frames.shape[0]):
                cv2.rectangle(frame, (int(coordinates[i-1, 0]), int(coordinates[i-1, 1])), (int(coordinates[i-1, 2]), int(coordinates[i-1, 3])), (255, 0, 0), 4)
                cv2.imshow("Frame", processed_frames[i])
        else:
            break

