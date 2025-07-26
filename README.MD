# Airwritten Burmese Alphabet Image Dataset

This dataset was created for an OCR system to recognize airwritten Burmese alphabets. It was developed as part of my 4th year Computer Vision project at UCSY (University of Computer Studies, Yangon).

## Dataset Details

- Purpose: OCR for airwritten Burmese alphabets
- Input Method: Finger tracking using webcam
- Format: RGB JPG images
- Resolution: 83 × 84 pixels

## Data Collection

To collect your own airwritten data, use the provided `capture.py` script. It uses MediaPipe and OpenCV to detect fingertip movement and draw on a black canvas. Press `c` to capture and save the image file and `d` to clear the canvas.
