import cv2
import numpy as np
import os


class ColorChallenge:
    """Clase para analizar regiones cromáticas en una imagen."""

    def __init__(self, image_name="original.jpg"):
        self.assets_path = "images"
        self.image_path = os.path.join(self.assets_path, image_name)
        self.image = None
        self.output_image = None

        if not os.path.exists(self.assets_path):
            os.makedirs(self.assets_path)

    def load_image(self):
        """Carga la imagen y muestra sus dimensiones."""
        self.image = cv2.imread(self.image_path)
        if self.image is None:
            raise FileNotFoundError(f"No se encontró la imagen en {self.image_path}")

        print(f"SISTEMA: Imagen cargada con éxito.")
        print(f"FORMA (Shape) de la imagen: {self.image.shape}")
        self.output_image = self.image.copy()

    def analyze_regions(self):
        """
        Define, analiza y dibuja las 6 regiones de interés (ROI).
        Formato de ROI: (x_inicio, y_inicio, ancho, alto)
        """
        rois = {
            "Region 1 (Rojo)": (100, 50, 30, 30),
            "Region 2 (Verde)": (150, 50, 30, 30),
            "Region 3 (Azul)": (250, 50, 30, 30),
            "Region 4 (Amarillo)": (50, 150, 30, 30),
            "Region 5 (Blanco)": (150, 150, 30, 30),
            "Region 6 (Mixta)": (250, 150, 50, 50)
        }

        print("\n" + "=" * 60)
        print(f"{'REGIÓN':<20} | {'CANAL':<8} | {'MEDIA':<10} | {'DESV. EST':<10}")
        print("-" * 60)

        for name, (x, y, w, h) in rois.items():
            # Extraer la región (Recorte de la matriz)
            roi = self.image[y:y + h, x:x + w]

            # Dibujar el recuadro en la imagen de salida
            cv2.rectangle(self.output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(self.output_image, name.split()[1], (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

            # Calcular estadísticas por cada canal (B, G, R)
            # axis=(0,1) calcula sobre los píxeles de la ROI
            means = np.mean(roi, axis=(0, 1))
            stds = np.std(roi, axis=(0, 1))

            channels = ['Azul (B)', 'Verde (G)', 'Rojo (R)']
            for i in range(3):
                print(f"{name:<20} | {channels[i]:<8} | {means[i]:<10.2f} | {stds[i]:<10.2f}")
            print("-" * 60)

    def show_results(self):
        output_path = os.path.join(self.assets_path, "resultado_rois.jpg")
        cv2.imwrite(output_path, self.output_image)

        cv2.imshow("Analisis de Colores - ROI", self.output_image)
        print(f"\nSISTEMA: Imagen de evidencia guardada en: {output_path}")
        print("Presione cualquier tecla para cerrar...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    try:
        challenge = ColorChallenge("original.jpg")
        challenge.load_image()
        challenge.analyze_regions()
        challenge.show_results()
    except Exception as e:
        print(f"ERROR: {e}")


if __name__ == "__main__":
    main()