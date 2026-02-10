import os

import cv2
import numpy as np


class ImageHandler:

    def __init__(self, subfolder="images"):
        # Ruta dinámica absoluta
        self.base_path = os.path.dirname(os.path.abspath(__file__))
        self.assets_path = os.path.join(self.base_path, subfolder)

        if not os.path.exists(self.assets_path):
            os.makedirs(self.assets_path)

    def load(self, file_name):
        path = os.path.join(self.assets_path, file_name)
        image = cv2.imread(path)
        if image is None:
            raise FileNotFoundError(f"No se pudo cargar la imagen en: {path}")
        return image

    def draw_roi(self, canvas, name, coords):
        x, y, w, h = coords
        label = name.split()[1]  # Extrae el color del nombre
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(canvas, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    def save(self, image, file_name):
        path = os.path.join(self.assets_path, file_name)
        cv2.imwrite(path, image)
        print(f"SISTEMA: Imagen de evidencia guardada en: {path}")


class ColorAnalyzer:

    @staticmethod
    def get_stats(roi):
        means = np.mean(roi, axis=(0, 1))
        stds = np.std(roi, axis=(0, 1))
        return means, stds


class App:

    def __init__(self, target_image="original.jpg"):
        self.handler = ImageHandler()
        self.analyzer = ColorAnalyzer()
        self.target_name = target_image

        # Diccionario de regiones de interés (x, y, w, h)
        self.rois = {
            "Region 1 (Amarillo)": (92, 320, 15, 15),
            "Region 2 (Verde)": (965, 635, 15, 15),
            "Region 3 (Azul)": (325, 360, 15, 15),
            "Region 4 (Blanco)": (140, 580, 15, 15),
            "Region 5 (Rojo)": (960, 500, 15, 15),
            "Region 6 (Mixta)": (520, 690, 40, 40)
        }

    def run(self):
        try:
            # 1. Carga
            original = self.handler.load(self.target_name)
            canvas = original.copy()
            print(f"SISTEMA: Imagen cargada con éxito. Shape: {original.shape}")

            print("\n" + "=" * 60)
            print(f"{'REGIÓN':<20} | {'CANAL':<8} | {'MEDIA':<10} | {'DESV. EST':<10}")
            print("-" * 60)

            for name, coords in self.rois.items():
                x, y, w, h = coords
                roi_pixels = original[y:y + h, x:x + w]

                means, stds = self.analyzer.get_stats(roi_pixels)

                self.handler.draw_roi(canvas, name, coords)

                channels = ['Azul (B)', 'Verde (G)', 'Rojo (R)']
                for i in range(3):
                    print(f"{name:<20} | {channels[i]:<8} | {means[i]:<10.2f} | {stds[i]:<10.2f}")
                print("-" * 60)

            # 3. Salida y Visualización
            self.handler.save(canvas, "resultado_rois.jpg")

            cv2.imshow("Analisis de Colores - ROI", canvas)
            print("SISTEMA: Presione cualquier tecla para cerrar...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        except Exception as e:
            print(f"ERROR: {e}")


def main():
    app = App("original.jpg")
    app.run()


if __name__ == "__main__":
    main()