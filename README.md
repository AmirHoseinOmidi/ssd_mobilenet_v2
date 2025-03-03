# Object Detection with SSD MobileNetV2

This project demonstrates real-time object detection using the SSD MobileNetV2 model, a state-of-the-art pre-trained deep learning model, in combination with TensorFlow and OpenCV. The model is capable of identifying and classifying a wide range of objects within a live video feed captured through a webcam. Detected objects are highlighted with bounding boxes, and their corresponding class names are displayed alongside the detection confidence scores.

The SSD (Single Shot Multibox Detector) MobileNetV2 model is optimized for mobile devices and provides a good balance between performance and speed. This project leverages this model to perform real-time object detection in a way that is efficient enough to run on a standard laptop or desktop.

### Key Features:
- Real-time object detection using webcam input.
- Bounding boxes drawn around detected objects with class names and confidence scores displayed.
- Performance optimized for smooth video streaming by limiting frame rate to 15 FPS.
- Supports detection of over 90 object classes, including people, vehicles, animals, and everyday items.

### Technologies Used:
- **TensorFlow**: A powerful open-source machine learning library used for loading and running the pre-trained SSD MobileNetV2 model. TensorFlow handles the model's inference and provides an easy-to-use interface for deploying deep learning models.
- **OpenCV**: OpenCV (Open Source Computer Vision Library) is used for video capture, real-time image processing, and rendering the output. OpenCV helps in manipulating frames from the webcam, resizing them to fit the model’s input size, and drawing the bounding boxes around detected objects.
- **NumPy**: NumPy is used for efficient handling of numerical data. It is particularly useful for processing image data, converting it to arrays, and manipulating detection outputs.
- **Time**: The `time` library is used to manage frame rate by calculating the time between frames, ensuring smooth operation of the video feed.

### How It Works:
The program continuously captures video frames from your webcam. These frames are resized to a 320x320 resolution, the input size required by the SSD MobileNetV2 model. Each resized frame is then passed through the model for inference. The model predicts bounding boxes, class labels, and confidence scores for objects detected in the frame.

- **Bounding Boxes**: The model returns a set of bounding boxes, each corresponding to a detected object. These boxes are normalized relative to the dimensions of the frame. The program then rescales these values to the original frame dimensions to draw the bounding boxes correctly.
- **Class Labels**: The class IDs returned by the model are mapped to human-readable class names, such as "person", "dog", or "car".
- **Confidence Scores**: Each prediction has a confidence score that indicates how confident the model is in its classification. The program filters out any predictions with a score below 0.5, focusing only on those with higher confidence.

The video feed is processed at a frame rate of 15 FPS, which can be adjusted based on your system’s performance. After processing each frame, the resulting image is displayed in a window where the detected objects are highlighted.

### Output:

Once you press the **'q'** key to stop the detection, the terminal will display a summary of the objects detected in the video feed, including the count of each object type detected and the average confidence score for each type. The output will look something like this:

```plaintext
Detection Finished. Results:
{
  "person": {"count": 3, "average_confidence": 0.85},
  "car": {"count": 2, "average_confidence": 0.92},
  "dog": {"count": 1, "average_confidence": 0.89},
  ...
}
‍‍‍‍```
. **Install Dependencies**:
   You need to install the required libraries. Run the following command to install TensorFlow, OpenCV, and NumPy:
   

bash
   pip install tensorflow opencv-python numpy
