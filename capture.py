import cv2
import mediapipe as mp
import numpy as np
import uuid
import os

resized   = None
crop_size = (83, 84)

def restart_canvas():
    # black canvas, contour_canvas
    # 3 is for channel
    return np.zeros((480, 640, 3), dtype=np.uint8), np.zeros((480, 640), dtype=np.uint8)    

vc = cv2.VideoCapture(index=0)

canvas, contour_canvas = restart_canvas()

# distance threshold between thumb tip and finger tip
threshold = 50

# bursh size of ink
brush_size = 5

# Stores previous finger position
prev_x, prev_y = None, None  

# draw mode
brush_mode = True

# Solutions API

# Inititalie MediaPipe FaceMesh 
mp_face_mesh = mp.solutions.face_mesh

# Initialize MediaPipe Hands
mp_hands   = mp.solutions.hands

# Instantiate the Hands class
hands = mp_hands.Hands(
    static_image_mode=False,       # False for video streams (better performance)
    max_num_hands=1,               # Max number of hands to detect
    min_detection_confidence=0.5,  # Confidence threshold to start tracking
    min_tracking_confidence=0.5    # Confidence threshold to continue tracking
)

# For drawing landmarks
mp_drawing = mp.solutions.drawing_utils  

while vc.isOpened():
    # read each frame
    success, frame = vc.read()

    # flip for mirror effect
    frame = cv2.flip(src=frame, flipCode=1)

    # media pipe requires RGB
    # works with RGB, but renders BGR
    rgb_frame = cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2RGB)

    # process with model
    detect = hands.process(rgb_frame)

    # frame height, width and number of channels (3)
    h, w, ch = frame.shape

    # if detected
    if detect.multi_hand_landmarks:
        for hand_landmarks in detect.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2),  # Landmark color
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)   # Connection color
            )

            # landmark node (x, y) is standarized to be [0, 1], 
            # so to get multiply it with frame values (finger.x * frame.x)
            finger_tip = hand_landmarks.landmark[8] 
            thumb_tip  = hand_landmarks.landmark[12]

            (finger_x, finger_y) = (int(finger_tip.x * w), int(finger_tip.y * h))
            (thumb_x, thumb_y)   = (int(thumb_tip.x * w), int(thumb_tip.y * h))

            # Compute Euclidean distance between index and thumb tips
            distance = np.linalg.norm(np.array([finger_x, finger_y]) - np.array([thumb_x, thumb_y]))
        
            if distance > threshold:

                cv2.circle(frame, (finger_x, finger_y), 5, (0, 255, 255), -1)  # Yellow preview circle
                
                # Draw on canvas if finger is moving
                if prev_x and prev_y:
                    cv2.line(canvas, (prev_x, prev_y), (finger_x, finger_y), (255, 255, 255), brush_size)
                
                prev_x, prev_y = finger_x, finger_y

            else:
                # red circle to show that currently not drawing
                cv2.circle(frame, (finger_x, finger_y), 5, (0, 0, 255), -1)
                
                if brush_mode: 
                    prev_x, prev_y = None, None

    # draw contours
    contours, _ = cv2.findContours(cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(contour_canvas, contours, -1, 255)

    if contours:
        x_min = min(cv2.boundingRect(c)[0] for c in contours)
        y_min = min(cv2.boundingRect(c)[1] for c in contours)
        x_max = max(cv2.boundingRect(c)[0] + cv2.boundingRect(c)[2] for c in contours)
        y_max = max(cv2.boundingRect(c)[1] + cv2.boundingRect(c)[3] for c in contours)

        # crop according to contour
        cropped = canvas.copy()[y_min:y_max, x_min:x_max]

        _, contour_canvas = restart_canvas()

        # cv2.imshow('cropped', cropped)

        # invert colors
        # cropped = cv2.bitwise_not(cropped)

        # resize to fit the desired size 
        resized = cv2.resize(cropped, crop_size)

        # if brush_mode: 
            # cropped = cv2.GaussianBlur(cropped, (15, 15), 0)

        # cv2.imshow('resized', resized)

    # overlay two images
    frame = cv2.addWeighted(frame, 0.6, canvas, 0.4, 0)

    cv2.putText(frame, f'threshold:{threshold}', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 1, cv2.LINE_AA)
    
    if brush_mode:
        mode = 'brush'
    else:
        mode = 'line'
    cv2.putText(frame, f'Mode:{mode}', (10,60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 1, cv2.LINE_AA)

    cv2.imshow('frame', frame)
    # cv2.imshow('canvas', canvas)
    # cv2.imshow('contour canvas', contour_canvas)

    key = cv2.waitKey(1)
    if key == 27: # esc
        break
    elif key == ord('m'): # m - 109
        threshold += 1
    elif key == ord('n'): # n - 110
        threshold -= 1
    elif key == ord('d'):
        canvas, contour_canvas = restart_canvas()
    elif key == ord('p'):
        brush_mode = not brush_mode
    elif key == ord('c'):
        # Generate a UUID and convert it to a string for use as a filename
        if 'saved' not in os.listdir('./'):
            os.makedirs('./saved', exist_ok=True)
        random_filename = str(uuid.uuid4())
        cv2.imwrite(f'./saved/{random_filename}.jpg', resized)

vc.release()
cv2.destroyAllWindows()