import random
import time
import os
import math
import csv
import matplotlib.pyplot as plt
import numpy as np


class RetoMatriz:
    def __init__(self, size=1000):
        self.size = size
        self.folder_data = "data"
        self.folder_imgs = "images"
        self.file_path = os.path.join(self.folder_data, "matriz_datos.csv")
        self.matriz = []

        # Crear carpetas si no existen
        for folder in [self.folder_data, self.folder_imgs]:
            if not os.path.exists(folder):
                os.makedirs(folder)

    def fase_1_nativo(self):
        print(">>> Iniciando Fase 1: Python Nativo")
        start_time = time.time()

        self.matriz = [[random.randint(0, 255) for _ in range(self.size)] for _ in range(self.size)]

        total_sum = 0
        total_sq_sum = 0
        min_val = 255
        max_val = 0
        n = self.size ** 2

        for fila in self.matriz:
            for val in fila:
                if val < min_val: min_val = val
                if val > max_val: max_val = val
                total_sum += val
                total_sq_sum += val ** 2

        media = total_sum / n
        varianza = (total_sq_sum / n) - (media ** 2)
        desv_est = math.sqrt(varianza)

        vector_aplanado = [val for fila in self.matriz for val in fila]
        with open(self.file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(vector_aplanado)

        end_time = time.time()
        duracion = end_time - start_time

        print(f"Fase 1 completada en: {duracion:.4f} segundos")
        return {"min": min_val, "max": max_val, "media": media, "std": desv_est, "tiempo": duracion}

    def fase_2_optimizado(self):
        print("\n>>> Iniciando Fase 2: Librerías Optimizadas (Numpy)")
        start_time = time.time()

        data_np = np.genfromtxt(self.file_path, delimiter=',')

        stats = {
            "min": np.min(data_np),
            "max": np.max(data_np),
            "media": np.mean(data_np),
            "std": np.std(data_np)
        }

        matriz_reconstruida = data_np.reshape((self.size, self.size))

        end_time = time.time()
        duracion = end_time - start_time

        print(f"Fase 2 completada en: {duracion:.4f} segundos")
        stats["tiempo"] = duracion
        return stats, matriz_reconstruida

    def mostrar_resultados(self, s1, s2, matriz_final):
        """Genera la imagen y muestra la comparativa."""
        print("\n" + "=" * 30)
        print("COMPARATIVA DE RESULTADOS")
        print("=" * 30)
        print(f"{'Métrica':<15} | {'Nativo':<12} | {'Numpy':<12}")
        print("-" * 45)
        for key in ["min", "max", "media", "std", "tiempo"]:
            print(f"{key:<15} | {s1[key]:<12.4f} | {s2[key]:<12.4f}")

        plt.figure(figsize=(10, 5))
        plt.imshow(matriz_final, cmap='viridis')
        plt.title(f"Imagen Generada ({self.size}x{self.size})")
        plt.colorbar()

        img_out = os.path.join(self.folder_imgs, "matriz_result.png")
        plt.savefig(img_out)
        print(f"\nImagen guardada en: {img_out}")
        plt.show()


if __name__ == "__main__":
    reto = RetoMatriz(1000)

    stats_nativas = reto.fase_1_nativo()
    stats_numpy, matriz_img = reto.fase_2_optimizado()

    reto.mostrar_resultados(stats_nativas, stats_numpy, matriz_img)