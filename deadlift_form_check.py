import cv2
import mediapipe as mp
import numpy as np
import os

# Setup Mediapipe instance
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Function to extract angles from keypoints (shoulder, hip, knee)
def extract_angles_from_keypoints(keypoints):
    # Reference body parts for calculation
    shoulder = np.array([keypoints[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                         keypoints[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y,
                         keypoints[mp_pose.PoseLandmark.LEFT_SHOULDER.value].z])
    
    hip = np.array([keypoints[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                    keypoints[mp_pose.PoseLandmark.LEFT_HIP.value].y,
                    keypoints[mp_pose.PoseLandmark.LEFT_HIP.value].z])
    
    knee = np.array([keypoints[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                     keypoints[mp_pose.PoseLandmark.LEFT_KNEE.value].y,
                     keypoints[mp_pose.PoseLandmark.LEFT_KNEE.value].z])

    # Calculate the angles between these points
    shoulder_hip_angle = calculate_angle(shoulder, hip, knee)  # Back angle
    hip_knee_angle = calculate_angle(hip, knee, np.array([knee[0], knee[1], knee[2] + 0.1]))  # Knee angle
    
    return shoulder_hip_angle, hip_knee_angle


# Function to calculate the angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)  # First point
    b = np.array(b)  # Middle point (elbow)
    c = np.array(c)  # Last point
    
    # Calculate vectors
    ab = a - b
    bc = c - b
    
    # Calculate angle using dot product
    dot = np.dot(ab, bc)
    cross = np.cross(ab, bc)
    angle = np.arctan2(np.linalg.norm(cross), dot)
    
    # Convert angle to degrees
    angle = np.degrees(angle)
    
    return angle


# Function to compare angles between user and reference with feedback
def compare_angles(reference_keypoints, user_keypoints, angle_threshold=5):
    # Extract angles from both reference and user keypoints
    reference_back, reference_knee = extract_angles_from_keypoints(reference_keypoints)
    user_back, user_knee = extract_angles_from_keypoints(user_keypoints)
    
    # Calculate the difference in angles
    back_diff = abs(reference_back - user_back)
    knee_diff = abs(reference_knee - user_knee)

    feedback = []

    # Provide feedback based on the angle differences and threshold
    if back_diff > angle_threshold:
        feedback.append(f"Your back angle is off by {back_diff:.2f} degrees. Try to keep your back straight.")
    if knee_diff > angle_threshold:
        feedback.append(f"Knee angle is off by {knee_diff:.2f} degrees. Make sure your knees are tracking over your toes.")
    
    if not feedback:
        feedback.append("Your form looks great! Keep it up.")

    return feedback


# Function to display feedback on the video frame
def display_feedback_on_video(frame, feedback):
    y_offset = 30  # Starting position for feedback text
    for msg in feedback:
        cv2.putText(frame, msg, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
        y_offset += 30  # Increase the y position for the next message


# Function to process video frames and provide feedback
def process_frame_and_provide_feedback(reference_video_path, user_video_path):
    # Load reference and user video
    cap_reference = cv2.VideoCapture(reference_video_path)
    cap_user = cv2.VideoCapture(user_video_path)
    
    if not cap_reference.isOpened() or not cap_user.isOpened():
        print("Error: Couldn't open video files.")
        return
    
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while True:
            # Read frames from both videos
            ret_reference, reference_frame = cap_reference.read()
            ret_user, user_frame = cap_user.read()
            
            if not ret_reference or not ret_user:
                break  # End of video

            # Convert frames to RGB
            reference_frame_rgb = cv2.cvtColor(reference_frame, cv2.COLOR_BGR2RGB)
            user_frame_rgb = cv2.cvtColor(user_frame, cv2.COLOR_BGR2RGB)
            
            # Process the frames with Mediapipe Pose
            results_reference = pose.process(reference_frame_rgb)
            results_user = pose.process(user_frame_rgb)

            if results_reference.pose_landmarks and results_user.pose_landmarks:
                reference_keypoints = results_reference.pose_landmarks.landmark
                user_keypoints = results_user.pose_landmarks.landmark

                # Compare the angles and get feedback
                feedback = compare_angles(reference_keypoints, user_keypoints)
                display_feedback_on_video(user_frame, feedback)  # Display feedback on the frame

            # Show the frame with feedback
            cv2.imshow('Deadlift Form Feedback', user_frame)

            # Stop if 'q' is pressed
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap_reference.release()
    cap_user.release()
    cv2.destroyAllWindows()


# Example usage (paths to video files)
reference_video_path = 'data/1128.mp4'  # Correct form reference video
user_video_path = 'data/test_video.mp4'  # User's video

process_frame_and_provide_feedback(reference_video_path, user_video_path)
