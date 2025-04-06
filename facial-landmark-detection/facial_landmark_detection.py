import cv2
import mediapipe as mp
import os
import argparse

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

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
        
        output_path = f"processed_{os.path.basename(image_path)}"
        cv2.imwrite(output_path, image)
        print(f"Processed image saved as {output_path}")

def process_webcam():
    """Process video from webcam in real-time."""
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:
        
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
            
            image = cv2.flip(image, 1)
            
            cv2.imshow('MediaPipe Face Mesh', image)
            
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
