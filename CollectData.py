import cv2
import mediapipe as mp
import time
import os

def Webcam2Landmarks(duration=1, frameCount=5):
    landmark_result = []
    img_result = []
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose
    cap = cv2.VideoCapture(0)
    with mp_pose.Pose(
        min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as pose:
        start_time = time.perf_counter()
        this_frame = 1
        time_gap = duration / frameCount
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image)

            # Draw the pose annotation on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

            # Flip the image horizontally for a selfie-view display.
            cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))

            if (results.pose_landmarks != None and (time.perf_counter() - start_time) >= time_gap * this_frame):
                landmark_result.append(results.pose_landmarks.landmark)
                img_result.append(image)
                this_frame += 1

            if this_frame > frameCount:
                cap.release()

            if cv2.waitKey(5) & 0xFF == 27:
                break
    return landmark_result, img_result


def WriteLandmarkss2File(file_path, landmarkss):
    file = open(file_path, "w")
    for landmarks in landmarkss:
        for landmark in landmarks:
            file.write(str(landmark.x))
            file.write(",")
            file.write(str(landmark.y))
            file.write(",")
            file.write(str(landmark.z))
            file.write(",")
        file.write("\n")

def CollectSamples(sample_count, pose, label):
    count = 1
    filename_list = os.listdir(os.path.join("Data/", pose, label))
    if len(filename_list) > 0:
        for i in range(len(filename_list)):
            filename_list[i] = int(filename_list[i][len(pose):-4])
        count = max(filename_list) + 1

    for i in range(sample_count):
        landmarkss, imgs = Webcam2Landmarks()
        WriteLandmarkss2File(os.path.join("Data/", pose, label, f"{pose}{count}.csv"), landmarkss)
        for i in range(5):
            try:
                os.makedirs(os.path.join("Images", pose, label, f"{pose}{count}"))
            except:
                pass
            file = open(os.path.join("Images", pose, label, f"{pose}{count}", f"{i + 1}.png"), "w")
            file.close()
            cv2.imwrite(os.path.join("Images", pose, label, f"{pose}{count}", f"{i + 1}.png"), imgs[i])
        count += 1


CollectSamples(10, "Smash", "positive")
