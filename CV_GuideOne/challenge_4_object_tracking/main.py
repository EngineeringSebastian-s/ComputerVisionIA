import cv2
import numpy as np
import tkinter as tk
from tkinter import messagebox
from pygrabber.dshow_graph import FilterGraph


class DetectorColor:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Configuración de Visión")
        self.root.geometry("400x200")

        self.color_ranges = {
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

    def obtener_lista_camaras(self):
        devices = FilterGraph().get_input_devices()
        return devices if devices else ["0"]

    def iniciar_interfaz(self):
        tk.Label(self.root, text="Seleccione el dispositivo de video:", font=("Arial", 10)).pack(pady=10)

        lista_camaras = self.obtener_lista_camaras()
        self.seleccion = tk.StringVar(self.root)
        self.seleccion.set(lista_camaras[0])  # Por defecto la primera

        dropdown = tk.OptionMenu(self.root, self.seleccion, *lista_camaras)
        dropdown.pack(pady=10)

        btn_iniciar = tk.Button(self.root, text="Iniciar Seguimiento",
                                command=lambda: self.ejecutar_procesamiento(lista_camaras))
        btn_iniciar.pack(pady=20)

        self.root.mainloop()

    def ejecutar_procesamiento(self, lista):
        idx = lista.index(self.seleccion.get())
        nombre_cam = self.seleccion.get()

        self.root.destroy()
        self.proceso_principal(idx, nombre_cam)

    def proceso_principal(self, index, nombre_cam):
        cap = cv2.VideoCapture(index)
        window_name = "Seguimiento - Camara: " + nombre_cam

        if not cap.isOpened():
            print("No se pudo abrir la cámara.")
            return

        while True:
            ret, frame = cap.read()
            if not ret: break

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            for color_name, ranges in self.color_ranges.items():
                mask = None
                for lower, upper in ranges:
                    current_mask = cv2.inRange(hsv, lower, upper)
                    mask = current_mask if mask is None else cv2.add(mask, current_mask)

                kernel = np.ones((5, 5), np.uint8)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    if cv2.contourArea(cnt) > 1500:
                        x, y, w, h = cv2.boundingRect(cnt)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(frame, f"{color_name}", (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow(window_name, frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    app = DetectorColor()
    app.iniciar_interfaz()