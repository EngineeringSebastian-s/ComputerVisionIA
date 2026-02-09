import tkinter as tk

import cv2
import numpy as np
from pygrabber.dshow_graph import FilterGraph


class CameraManager:

    @staticmethod
    def get_available_cameras():
        devices = FilterGraph().get_input_devices()
        return devices if devices else ["0"]


class ColorProcessor:
    def __init__(self):
        # Definición de rangos HSV para los colores
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
        self.kernel = np.ones((5, 5), np.uint8)

    def process_frame(self, frame):
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        for color_name, ranges in self.color_ranges.items():
            mask = None
            for lower, upper in ranges:
                current_mask = cv2.inRange(hsv_frame, lower, upper)
                mask = current_mask if mask is None else cv2.add(mask, current_mask)

            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                if cv2.contourArea(cnt) > 1500:
                    x, y, w, h = cv2.boundingRect(cnt)
                    # Dibujar rectángulo y etiqueta
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, f"{color_name}", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        return frame


class ChallengeFour:

    def __init__(self):
        self.selected_camera = None
        self.camera_list = None
        self.processor = ColorProcessor()
        self.camera_manager = CameraManager()
        self.root = tk.Tk()
        self.setup_gui()

    def setup_gui(self):
        self.root.title("Configuración de Visión")
        self.root.geometry("400x220")

        tk.Label(self.root, text="--- Detector de Colores ---", font=("Arial", 12, "bold")).pack(pady=10)
        tk.Label(self.root, text="Seleccione el dispositivo de video:", font=("Arial", 10)).pack(pady=5)

        self.camera_list = self.camera_manager.get_available_cameras()
        self.selected_camera = tk.StringVar(self.root)
        self.selected_camera.set(self.camera_list[0])

        dropdown = tk.OptionMenu(self.root, self.selected_camera, *self.camera_list)
        dropdown.pack(pady=10)

        btn_start = tk.Button(self.root, text="INICIAR CÁMARA", bg="#2ecc71", fg="white",
                              command=self.start_vision_loop, font=("Arial", 10, "bold"))
        btn_start.pack(pady=15)

    def start_vision_loop(self):
        camera_name = self.selected_camera.get()
        camera_index = self.camera_list.index(camera_name)

        self.root.destroy()
        self.run_opencv_thread(camera_index, camera_name)

    def run_opencv_thread(self, index, name):
        cap = cv2.VideoCapture(index)
        window_title = f"Camara: {name} (ESC para salir)"

        print(f"SISTEMA: Iniciando captura en '{name}'...")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("ERROR: No se pudo obtener el frame de la cámara.")
                break

            processed_frame = self.processor.process_frame(frame)

            cv2.imshow(window_title, processed_frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break
            if cv2.getWindowProperty(window_title, cv2.WND_PROP_VISIBLE) < 1:
                break

        cap.release()
        cv2.destroyAllWindows()
        print("SISTEMA: Proceso finalizado por el usuario.")


def main():
    app = ChallengeFour()
    app.root.mainloop()


if __name__ == "__main__":
    main()
