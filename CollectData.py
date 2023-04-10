import cv2
import mediapipe as mp
import time
import os
import winsound

def Webcam2LandmarksDynamic(duration=2, frameCount=8):
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
        winsound.Beep(500, 500)
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

            if (results.pose_landmarks != None and (time.perf_counter() - start_time) >= time_gap * this_frame):
                mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

                # Flip the image horizontally for a selfie-view display.
                cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
                landmark_result.append(results.pose_landmarks.landmark)
                img_result.append(image)
                this_frame += 1

            if this_frame > frameCount:
                cap.release()

            if cv2.waitKey(5) & 0xFF == 27:
                break
    return landmark_result, img_result


def Webcam2LandmarksStatic():
    landmark_result = []
    img_result = []
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose
    cap = cv2.VideoCapture(0)
    with mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as pose:
        winsound.Beep(500, 500)
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

            if (results.pose_landmarks != None):
                landmark_result.append(results.pose_landmarks.landmark)
                img_result.append(image)
                cap.release()

            if cv2.waitKey(5) & 0xFF == 27:
                break
    return landmark_result, img_result


def WriteLandmarkss2File(file_path, landmarkss):
    file = open(file_path, "w")
    for landmarks in landmarkss:
        cnt = 0
        for landmark in landmarks:
            file.write(str(landmark.x))
            file.write(",")
            file.write(str(landmark.y))
            file.write(",")
            file.write(str(landmark.z))
            if cnt != 32:
                file.write(",")
            cnt += 1
        file.write("\n")


def CollectSamples(sample_count, pose, label, mode):

    count = 1
    try:
        os.makedirs(os.path.join(mode + "Data/", pose, label))
    except:
        pass
    filename_list = os.listdir(os.path.join(mode + "Data/", pose, label))
    if len(filename_list) > 0:
        for i in range(len(filename_list)):
            filename_list[i] = int(filename_list[i][len(pose):-4])
        count = max(filename_list) + 1

    for i in range(sample_count):
        if mode == "Dynamic":
            landmarkss, imgs = Webcam2LandmarksDynamic()
        else:
            landmarkss, imgs = Webcam2LandmarksStatic()

        WriteLandmarkss2File(os.path.join(mode + "Data/", pose, label, f"{pose}{count}.csv"), landmarkss)
        for j in range(8 if mode == "Dynamic" else 1):
            try:
                os.makedirs(os.path.join(mode + "Images", pose, label, f"{pose}{count}"))
            except:
                pass
            file = open(os.path.join(mode + "Images", pose, label, f"{pose}{count}", f"{j + 1}.png"), "w")
            file.close()
            cv2.imwrite(os.path.join(mode + "Images", pose, label, f"{pose}{count}", f"{j + 1}.png"), imgs[j])
        count += 1


CollectSamples(5, "Smash", "negative", "Dynamic")
