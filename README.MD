# Air-written Burmese Alphabet Image Dataset

This dataset was created for an OCR system to recognize air-written Burmese alphabets. It was developed as part of my 4th year Computer Vision project at UCSY (University of Computer Studies, Yangon).

## Dataset Details

- Purpose: OCR for air-written Burmese alphabets
- Input Method: Finger tracking using webcam
- Format: **BGR** JPG images
- Resolution: 83 × 84 pixels

This dataset includes all Burmese alphabets recorded in 83×84 resolution, with 30 images per character.
Some characters are also available in 63×64 resolution.
An additional folder (extra/) contains more 83×84 images for some characters.

The dataset was created manually using a custom Python application with OpenCV and MediaPipe for real-time finger tracking via webcam. Each character was written in the air by a human subject using their index finger. The fingertip position was tracked and visualized on a virtual canvas. The resulting strokes are cropped tightly around the drawn character using contour detection and resized to 83x84 BGR .jpg images.

## Data Collection

To collect your own air-written data, use the provided `capture.py` script. It uses MediaPipe and OpenCV to detect fingertip movement and draw on a black canvas. Press `c` to capture and save the image file and `d` to clear the canvas.
