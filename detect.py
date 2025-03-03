import cv2
import tensorflow as tf
import numpy as np
import time

model_dir = '/home/amir/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/saved_model'
model = tf.saved_model.load(model_dir)

def run_inference_for_single_image(model, image):
    image = np.asarray(image)
    input_tensor = tf.convert_to_tensor(image, dtype=tf.uint8)
    input_tensor = input_tensor[tf.newaxis, ...]

    model_fn = model.signatures["serving_default"]
    output_dict = model_fn(input_tensor)

    output_dict = {key: value.numpy() for key, value in output_dict.items()}
    return output_dict

class_names = ["0.background", "1. person", "2. bicycle", "3. car", "4. motorbike", "5. airplane", "6. bus", "7. train", "8. truck", "9. boat", 
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
"88. teddybear", "89. hairdrier", "90. toothbrush", "91. hairbrush"]

cap = cv2.VideoCapture(2)

fps_limit = 15
frame_interval = int(1000 / fps_limit)

detected_objects = []  # لیستی برای ذخیره اشیاء شناسایی‌شده

while True:
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    image_resized = cv2.resize(frame, (320, 320))

    output_dict = run_inference_for_single_image(model, image_resized)

    boxes = output_dict.get('detection_boxes', [])
    classes = output_dict.get('detection_classes', [])
    scores = output_dict.get('detection_scores', [])

    if len(boxes) > 0:
        boxes = boxes[0]
        classes = classes[0].astype(int)
        scores = scores[0]

        for i in range(len(boxes)):
            if scores[i] > 0.5:  
                y1, x1, y2, x2 = boxes[i]

                (startX, startY, endX, endY) = (int(x1 * frame.shape[1]), int(y1 * frame.shape[0]),
                int(x2 * frame.shape[1]), int(y2 * frame.shape[0]))

                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

                class_id = classes[i]
                class_name = class_names[class_id] if class_id < len(class_names) else "Unknown"
                label = f"{class_name}: {scores[i]:.2f}"
                cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                detected_objects.append(f"{class_name}: {scores[i]:.2f}")

    cv2.imshow('Object Detection', frame)

    elapsed_time = time.time() - start_time
    sleep_time = frame_interval - (elapsed_time * 1000)
    if sleep_time > 0:
        time.sleep(sleep_time / 1000)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("Detection Finished. Results:")
for obj in detected_objects:
    print(obj)
