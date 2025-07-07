import socket
import cv2
import numpy as np
import tensorflow as tf
from threading import Thread

# COCO class labels
class_names = [
    "0.background", "1. person", "2. bicycle", "3. car", "4. motorbike", "5. airplane", "6. bus", "7. train", "8. truck", "9. boat", 
    "10. trafficlight", "11. firehydrant", "12. streetsign", "13. stopsign", "14. parkingmeter", "15. bench", "16. bird", 
    "17. cat", "18. dog", "19. horse", "20. sheep", "21. cow", "22. elephant", "23. bear", "24. zebra", "25. giraffe", 
    "26. hat", "27. backpack", "28. umbrella", "29. shoe", "30. eyeglasses", "31. handbag", "32. tie", "33. suitcase", 
    "34. frisbee", "35. skis", "36. snowboard", "37. sportsball", "38. kite", "39. baseballbat", "40. baseballglove", 
    "41. skateboard", "42. surfboard", "43. tennisracket", "44. bottle", "45. plate", "46. wineglass", "47. cup", 
    "48. fork", "49. knife", "50. spoon", "51. bowl", "52. banana", "53. apple", "54. sandwich", "55. orange", 
    "56. broccoli", "57. carrot", "58. hotdog", "59. pizza", "60. donut", "61. cake", "62. chair", "63. sofa", 
    "64. pottedplant", "65. bed", "66. mirror", "67. diningtable", "68. window", "69. desk", "70. toilet", "71. door", 
    "72. tvmonitor", "73. laptop", "74. mouse", "75. remote", "76. keyboard", "77. cellphone", "78. microwave", 
    "79. oven", "80. toaster", "81.sink", "82. refrigerator", "83. blender", "84. book", "85. clock", "86. vase", "87. scissors",
    "88. teddybear", "89. hairdrier", "90. toothbrush", "91. hairbrush"
]

model = tf.saved_model.load('/home/amir/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/saved_model')
model_fn = model.signatures['serving_default']

def process_detection(image):
    # running model and return a dict of tensor objects
    input_tensor = tf.convert_to_tensor(image, dtype=tf.uint8)[tf.newaxis, ...]
    output_dict = model_fn(input_tensor)
    return output_dict

def handle_client(conn, addr):
    # managing client
    print(f"[SERVER] Connection from {addr}")
    try:
        while True:
            # collecting data length
            length_bytes = conn.recv(4)
            if not length_bytes:
                break
            length = int.from_bytes(length_bytes, byteorder='big')
            
            # collecting image data
            data = b''
            while len(data) < length:
                packet = conn.recv(min(length - len(data), 4096))
                if not packet:
                    break
                data += packet
            
            if len(data) != length:
                print(f"[SERVER] Incomplete data from {addr}")
                break
            
            # Image processing
            nparr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None:
                print(f"[SERVER] Failed to decode image from {addr}")
                continue
            
            # running model
            resized = cv2.resize(frame, (320, 320))
            detections = process_detection(resized)
            
            # Draw the results on the screen
            boxes = detections['detection_boxes'][0].numpy()
            scores = detections['detection_scores'][0].numpy()
            classes = detections['detection_classes'][0].numpy().astype(int)
            
            for i in range(min(10, len(scores))):  # at most 10 objects
                if scores[i] > 0.5:  # Confidence threshold
                    h, w = frame.shape[:2]
                    y1, x1, y2, x2 = boxes[i]
                    x1, y1, x2, y2 = int(x1*w), int(y1*h), int(x2*w), int(y2*h)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{class_names[classes[i]]}: {scores[i]:.2f}"
                    cv2.putText(frame, label, (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Send processed image
            _, jpeg = cv2.imencode('.jpg', frame)
            out_bytes = jpeg.tobytes()

            conn.send(len(out_bytes).to_bytes(4, byteorder='big'))
            conn.send(out_bytes)
            
    except Exception as e:
        print(f"[SERVER] Error with {addr}: {e}")
    finally:
        conn.close()
        print(f"[SERVER] Connection with {addr} closed")

def start_server():
    # running server
    server = socket.socket()
    server.bind(('0.0.0.0', 9999))
    server.listen(5)
    print("[SERVER] Waiting for connections...")
    
    try:
        while True:
            conn, addr = server.accept()
            client_thread = Thread(target=handle_client, args=(conn, addr))
            client_thread.daemon = True
            client_thread.start()
    except KeyboardInterrupt:
        print("\n[SERVER] Shutting down...")
    finally:
        server.close()

if __name__ == "__main__":
    start_server()
