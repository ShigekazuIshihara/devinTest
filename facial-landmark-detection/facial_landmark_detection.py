import cv2
import mediapipe as mp
import os
import argparse
import numpy as np
import math

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

LEFT_EYE_UPPER = [159, 386]  # Upper eyelid landmarks
LEFT_EYE_LOWER = [145, 374]  # Lower eyelid landmarks
LEFT_IRIS = [468, 469, 470, 471, 472]  # Iris landmarks

RIGHT_EYE_UPPER = [386, 159]  # Upper eyelid landmarks
RIGHT_EYE_LOWER = [374, 145]  # Lower eyelid landmarks
RIGHT_IRIS = [473, 474, 475, 476, 477]  # Iris landmarks

def calculate_distance(point1, point2):
    """Calculate Euclidean distance between two points."""
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2 + (point1.z - point2.z)**2)

def calculate_iris_diameter(landmarks, iris_points):
    """Calculate the diameter of the iris using iris landmarks."""
    left_point = landmarks.landmark[iris_points[0]]
    right_point = landmarks.landmark[iris_points[2]]
    return calculate_distance(left_point, right_point)

def calculate_eye_opening(landmarks, upper_points, lower_points):
    """Calculate the vertical distance between eyelids."""
    distances = []
    for upper, lower in zip(upper_points, lower_points):
        upper_point = landmarks.landmark[upper]
        lower_point = landmarks.landmark[lower]
        distances.append(calculate_distance(upper_point, lower_point))
    return sum(distances) / len(distances)

def calculate_perclos(eye_opening, iris_diameter):
    """Calculate PERCLOS (percentage of eye closure)."""
    if iris_diameter == 0:  # Avoid division by zero
        return 0
    
    percentage = (eye_opening / iris_diameter) * 100 - 90 # This minus CONSTANT parameter is depends on the subject.
    return percentage

def is_eye_closed(perclos_value, threshold=20):
    """Determine if the eye is closed based on PERCLOS value."""
    return perclos_value < threshold

def process_image(image_path):
    """Process a single image file."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image from {image_path}")
        return
    
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5) as face_mesh:
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        results = face_mesh.process(image_rgb)
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
                
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
                
                if hasattr(mp_face_mesh, 'FACEMESH_IRISES'):
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_IRISES,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style())
                
                left_iris_diameter = calculate_iris_diameter(face_landmarks, LEFT_IRIS)
                right_iris_diameter = calculate_iris_diameter(face_landmarks, RIGHT_IRIS)
                
                left_eye_opening = calculate_eye_opening(face_landmarks, LEFT_EYE_UPPER, LEFT_EYE_LOWER)
                right_eye_opening = calculate_eye_opening(face_landmarks, RIGHT_EYE_UPPER, RIGHT_EYE_LOWER)
                
                left_perclos = calculate_perclos(left_eye_opening, left_iris_diameter)
                right_perclos = calculate_perclos(right_eye_opening, right_iris_diameter)
                
                left_eye_closed = is_eye_closed(left_perclos)
                right_eye_closed = is_eye_closed(right_perclos)
                
                cv2.putText(image, f"LEFT Eye PERCLOS: {left_perclos:.2f}%", (10, 30),  
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(image, f"RIGHT Eye PERCLOS: {right_perclos:.2f}%", (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                if left_eye_closed:
                    cv2.putText(image, "LEFT Eye Closed", (10, 90),  
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                if right_eye_closed:
                    cv2.putText(image, "RIGHT Eye Closed", (10, 120), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        output_path = f"processed_{os.path.basename(image_path)}"
        cv2.imwrite(output_path, image)
        print(f"Processed image saved as {output_path}")

def process_webcam():
    """Process video from webcam in real-time."""
    cap = cv2.VideoCapture(1)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:
        
        left_perclos_history = []
        right_perclos_history = []
        history_size = 10  # Number of frames to average over
        
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue
            
            image.flags.writeable = False
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            results = face_mesh.process(image_rgb)
            
            image.flags.writeable = True
            
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
                    
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
                    
                    if hasattr(mp_face_mesh, 'FACEMESH_IRISES'):
                        mp_drawing.draw_landmarks(
                            image=image,
                            landmark_list=face_landmarks,
                            connections=mp_face_mesh.FACEMESH_IRISES,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style())
                    
                    left_iris_diameter = calculate_iris_diameter(face_landmarks, LEFT_IRIS)
                    right_iris_diameter = calculate_iris_diameter(face_landmarks, RIGHT_IRIS)
                    
                    left_eye_opening = calculate_eye_opening(face_landmarks, LEFT_EYE_UPPER, LEFT_EYE_LOWER)
                    right_eye_opening = calculate_eye_opening(face_landmarks, RIGHT_EYE_UPPER, RIGHT_EYE_LOWER)
                    
                    left_perclos = calculate_perclos(left_eye_opening, left_iris_diameter)
                    right_perclos = calculate_perclos(right_eye_opening, right_iris_diameter)
                    
                    left_perclos_history.append(left_perclos)
                    right_perclos_history.append(right_perclos)
                    
                    if len(left_perclos_history) > history_size:
                        left_perclos_history.pop(0)
                    if len(right_perclos_history) > history_size:
                        right_perclos_history.pop(0)
                    
                    avg_left_perclos = sum(left_perclos_history) / len(left_perclos_history)
                    avg_right_perclos = sum(right_perclos_history) / len(right_perclos_history)
                    
                    left_eye_closed = is_eye_closed(avg_left_perclos)
                    right_eye_closed = is_eye_closed(avg_right_perclos)
                    
                    cv2.putText(image, f"left Eye PERCLOS: {avg_left_perclos:.2f}%", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(image, f"right Eye PERCLOS: {avg_right_perclos:.2f}%", (10, 60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    if left_eye_closed:
                        cv2.putText(image, "left Eye Closed", (10, 90),  
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    if right_eye_closed:
                        cv2.putText(image, "right Eye Closed", (10, 120), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    h, w, _ = image.shape
                    cv2.rectangle(image, (w-150, 30), (w-50, 50), (255, 255, 255), 2)
                    eye_bar_width = int(avg_left_perclos)
                    if eye_bar_width > 100:
                        eye_bar_width = 100
                    cv2.rectangle(image, (w-150, 30), (w-150+eye_bar_width, 50), 
                                 (0, 255, 0) if not left_eye_closed else (0, 0, 255), -1)
                    
                    cv2.rectangle(image, (w-150, 60), (w-50, 80), (255, 255, 255), 2)
                    eye_bar_width = int(avg_right_perclos)
                    if eye_bar_width > 100:
                        eye_bar_width = 100
                    cv2.rectangle(image, (w-150, 60), (w-150+eye_bar_width, 80), 
                                 (0, 255, 0) if not right_eye_closed else (0, 0, 255), -1)
            
            image = cv2.flip(image, 1)
            
            cv2.imshow('MediaPipe Face Mesh with PERCLOS', image)
            
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
    
    cap.release()
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='Facial Landmark Detection using MediaPipe')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--webcam', action='store_true', help='Use webcam for real-time detection')
    group.add_argument('--image', type=str, help='Path to image file for processing')
    
    args = parser.parse_args()
    
    if args.webcam:
        print("Starting real-time facial landmark detection using webcam...")
        process_webcam()
    elif args.image:
        print(f"Processing image: {args.image}")
        process_image(args.image)

if __name__ == '__main__':
    main()
