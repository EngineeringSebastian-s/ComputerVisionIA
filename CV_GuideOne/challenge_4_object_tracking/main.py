import cv2
import numpy as np

def seleccionar_camara():
    print("--- Configuración de Cámara ---")
    try:
        index = int(input("Ingrese el índice de la cámara (usualmente 0 para la integrada): "))
        return index
    except ValueError:
        print("Entrada no válida. Usando cámara 0 por defecto.")
        return 0

def procesar_seguimiento():
    cam_index = seleccionar_camara()
    cap = cv2.VideoCapture(cam_index)

    if not cap.isOpened():
        print("Error: No se pudo acceder a la cámara.")
        return

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

    window_name = "Seguimiento de figura por color"
    cv2.namedWindow(window_name)

    print("Iniciando... Presiona ESC o cierra la ventana para salir.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        for color_name, ranges in color_ranges.items():
            mask = None
            for lower, upper in ranges:
                current_mask = cv2.inRange(hsv, lower, upper)
                mask = current_mask if mask is None else cv2.add(mask, current_mask)

            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                if cv2.contourArea(cnt) > 1000:
                    x, y, w, h = cv2.boundingRect(cnt)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, f"Color: {color_name}", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow(window_name, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Programa finalizado correctamente.")

if __name__ == "__main__":
    procesar_seguimiento()