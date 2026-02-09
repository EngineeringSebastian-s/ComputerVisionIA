import cv2
import numpy as np

cap = cv2.VideoCapture(0)

color_ranges = {
    "Rojo": [
        (np.array([0, 120, 70]), np.array([10, 255, 255])),
        (np.array([170, 120, 70]), np.array([180, 255, 255]))
    ],
    "Verde": [
        (np.array([35, 100, 100]), np.array([85, 255, 255]))
    ],
    "Azul": [
        (np.array([100, 150, 0]), np.array([140, 255, 255]))
    ]
}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    for color_name, ranges in color_ranges.items():
        mask = None

        for lower, upper in ranges:
            current_mask = cv2.inRange(hsv, lower, upper)
            mask = current_mask if mask is None else mask + current_mask

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 1000:  
                x, y, w, h = cv2.boundingRect(cnt)

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                cv2.putText(
                    frame,
                    f"Color: {color_name}",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

    cv2.imshow("Seguimiento de figura por color", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()