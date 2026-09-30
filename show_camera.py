import cv2
import time
from flask import Flask, Response

app = Flask(__name__)

def generate_frames():
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    # Снижаем разрешение до 640x480
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Даем подсистеме V4L2 время на инициализацию
    time.sleep(1.0) 

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01) # Ждем, если кадр не успел прийти
            continue
            
        # Конвертируем кадр в формат JPEG
        success, buffer = cv2.imencode('.jpg', frame)
        if not success:
            continue
            
        frame_bytes = buffer.tobytes()
        
        # Отправляем кадр в браузер
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    print("Трансляция (640x480) запущена! Откройте в браузере: http://IP_ВАШЕЙ_МАЛИНКИ:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)