import cv2
import numpy as np
import sys, getopt
from kalmanfilter import KalmanFilter

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
def find_best_bb(old_frame, new_frame, x, y, x1, y1, gamma=0.03):
    ref_img = old_frame[y:y1+1, x:x1+1]

    W = old_frame.shape[1]
    H = old_frame.shape[0]

    W_bb = x1 - x
    H_bb = y1 - y

    # find search bounds within gamma % of old bb
    low_Xi = max(0, int(x * (1 - gamma)))
    high_Xi = min(W - 1, int(x * (1 + gamma)))

    low_Yj = max(0, int(y * (1 - gamma)))
    high_Yj = min(H - 1, int(y * (1 + gamma)))

    # do quadruple for loop, checking area is within 10% of previous, and constraint that u > i, and v > j
    # i, j are upper left, u, v are lower right

    best_correlation = 0
    best_offset = (x, y, x1, y1)

    max_x = new_frame.shape[1]
    max_y = new_frame.shape[0]

    for i in range(low_Xi, high_Xi):
        for j in range(low_Yj, high_Yj):
            # resize image to make same with ref, old frame
            dim = (ref_img.shape[1], ref_img.shape[0])

            edge_x = i + W_bb
            edge_y = j + H_bb

            if edge_x > max_x:
                edge_x = max_x
            if edge_y > max_y:
                edge_y = max_y

            temp_ref = new_frame[j:edge_y+1, i:edge_x+1]
            if temp_ref.shape != ref_img.shape:
                try:
                    temp_ref = cv2.resize(new_frame[j:edge_y+1, i:edge_x+1], dim, interpolation=cv2.INTER_AREA)
                except:
                    print("error resizing")
                    continue
            # compute correlation
            corr = normalized_cross_correlation(temp_ref, ref_img)

            # update best correlation and bounding box
            if (corr > best_correlation):
                best_correlation = corr

                best_offset = (i, j, i + W_bb, j + H_bb)
                        

    return best_offset    

if __name__ == "__main__":
    gamma = 0.03

    argList = sys.argv[1:]
    options = "g:"
    long_options = ["gamma="]

    arguments, _ = getopt.getopt(argList, options, long_options)
    for arg, val in arguments:
        if arg == "--gamma":
            gamma = val
        elif arg == "-g":
            gamma = float(val[1:])
    
    vid_name = input("Enter video file:\n")
    mode = input("Play frames while processing? y or n\n")

    if mode == "y":
        playback = True
    else:
        playback = False

    cap = cv2.VideoCapture(vid_name)
    num_frames = int(cap.get(7))

    # Load Kalman filter to predict the trajectory
    kf = KalmanFilter()

    print("Processing video...")

    iter = 0
    initial_coordinates = vid_name.split("_")
    x = int(initial_coordinates[0])
    y = int(initial_coordinates[1])
    x1 = int(initial_coordinates[2])
    y1 = initial_coordinates[3]
    y1 = int(y1[:y1.find(".")])

    processed_frames = []
    coordinates = []

    while True:
        ret, frame = cap.read()

        if ret is False:
            print(f"Completed processing {iter}/{num_frames} frames.")
            break

        if iter > 0:
            x, y, x1, y1 = find_best_bb(old_frame, frame, x, y, x1, y1, gamma=gamma)

            cx = int((x + x1) / 2)
            cy = int((y + y1) / 2)

            width = abs(x1 - x)
            height = abs(y1 - y)

            predicted = kf.predict(cx, cy)
            # print("PREDICTED", predicted)

            px = abs(predicted[0] - width / 2)
            px1 = abs(predicted[0] + width / 2)
            py = abs(predicted[1] - height / 2)
            py1 = abs(predicted[1] + height / 2)

            predicted = np.array([px, py, px1, py1])
            coordinates.append(predicted)
        else:
            cx = int((x + x1) / 2)
            cy = int((y + y1) / 2)
            kf.predict(cx, cy)

        if iter % 7 == 0 and iter != 0:
            print(f"Processing frame {iter+1}/{num_frames}...")

        old_frame = frame
        processed_frames.append(old_frame)
        
        if playback:
            cv2.rectangle(frame, (int(x), int(y)), (int(x1), int(y1)), (255, 0, 0), 4)

            cv2.imshow("Frame", frame)
            key = cv2.waitKey(11)

            if key == 27:
                break
        
        iter += 1
    
    processed_frames = np.array(processed_frames)
    coordinates = np.array(coordinates)
    
    print("Finished processing video. Saving...")

    box_frames = []
    box_frames.append(processed_frames[0])
    for i in range(1, processed_frames.shape[0]):
        box = cv2.rectangle(processed_frames[i], (int(coordinates[i-1, 0]), int(coordinates[i-1, 1])), (int(coordinates[i-1, 2]), int(coordinates[i-1, 3])), (255, 0, 0), 4)
        box_frames.append(box)
    box_frames = np.array(box_frames)

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(5))
    cap.release()

    output = cv2.VideoWriter(f"output_{vid_name}", cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    for frame in box_frames:
        output.write(frame)
    output.release()

    print("Video saved.")
