from keras.models import load_model
from keras.layers import DepthwiseConv2D
from PIL import Image, ImageOps
import numpy as np
import cv2
import paho.mqtt.client as mqtt

# Callback function for connection
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected to Adafruit IO!")
    else:
        print("Failed to connect, return code %d\n", rc)



# Tùy chỉnh DepthwiseConv2D để tải mô hình nếu cần
def custom_depthwise_conv2d(*args, **kwargs):
    kwargs.pop('groups', None)
    return DepthwiseConv2D(*args, **kwargs)

custom_objects = {'DepthwiseConv2D': custom_depthwise_conv2d}

# Tải mô hình
model = load_model('keras_model.h5', custom_objects=custom_objects)

# Hàm xử lý hình ảnh
def process_image(file_path):
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    image = Image.open(file_path).convert("RGB")
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    # Chuyển đổi và chuẩn hóa dữ liệu ảnh
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data[0] = normalized_image_array

    # Dự đoán bằng mô hình
    prediction = model.predict(data)
    output = prediction[0]

    # Tìm lớp có độ tin cậy cao nhất
    max_index = np.argmax(output)
    max_confidence = output[max_index]

    # Đọc nhãn từ labels.txt
    with open('labels.txt', encoding="utf8") as file:
        labels = file.read().split("\n")

    result = f"AI Result: {labels[max_index]} (Confidence: {max_confidence:.2f})"
    return result

# Mở camera
def open_camera():
    cam = cv2.VideoCapture(0)

    if not cam.isOpened():
        print("Không thể mở camera!")
        return

    print("Nhấn 'C' để chụp ảnh, 'Q' để thoát.")
    image_count = 0  # Đếm số ảnh chụp

    while True:
        ret, frame = cam.read()
        if not ret:
            print("Không thể đọc frame từ camera!")
            break

        # Hiển thị khung hình từ camera
        cv2.imshow("Camera", frame)

        # Bắt sự kiện phím
        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):  # Nhấn 'C' để chụp ảnh
            image_count += 1
            file_path = f"captured_image_{image_count}.jpg"  # Lưu tên tệp khác nhau mỗi lần chụp
            cv2.imwrite(file_path, frame)
            print(f"Ảnh đã được chụp và lưu thành '{file_path}'.")

            # Dự đoán hình ảnh
            result = process_image(file_path)
            print(result)
            data = result[10:] # send to mqtt

        elif key == ord('q'):  # Nhấn 'Q' để thoát
            print("Đang thoát...")
            break

    # Đóng camera và cửa sổ
    cam.release()
    cv2.destroyAllWindows()

# Gọi hàm mở camera
open_camera()
