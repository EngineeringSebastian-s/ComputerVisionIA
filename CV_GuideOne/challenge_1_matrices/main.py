import csv
import math
import os
import random
import time
import tkinter as tk
from tkinter import messagebox

import matplotlib.pyplot as plt
import numpy as np


class MatrixProcessor:

    def __init__(self, size=1000):
        self.size = size
        self.folder_data = "data"
        self.folder_imgs = "images"
        self.file_path = os.path.join(self.folder_data, "matrix_data.csv")
        self.matrix = []

        for folder in [self.folder_data, self.folder_imgs]:
            if not os.path.exists(folder):
                os.makedirs(folder)

    def run_native_processing(self):
        start_time = time.time()

        self.matrix = [[random.randint(0, 255) for _ in range(self.size)] for _ in range(self.size)]

        total_sum = 0
        total_sq_sum = 0
        min_val = 255
        max_val = 0
        n = self.size ** 2

        for row in self.matrix:
            for val in row:
                if val < min_val: min_val = val
                if val > max_val: max_val = val
                total_sum += val
                total_sq_sum += val ** 2

        mean_val = total_sum / n
        variance = (total_sq_sum / n) - (mean_val ** 2)
        std_dev = math.sqrt(variance)

        flat_vector = [val for row in self.matrix for val in row]
        with open(self.file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(flat_vector)

        duration = time.time() - start_time
        return {"min": min_val, "max": max_val, "media": mean_val, "std": std_dev, "time": duration}

    def run_numpy_processing(self):
        start_time = time.time()

        data_np = np.genfromtxt(self.file_path, delimiter=',')

        stats = {
            "min": np.min(data_np),
            "max": np.max(data_np),
            "media": np.mean(data_np),
            "std": np.std(data_np)
        }

        matrix_reconstructed = data_np.reshape((self.size, self.size))
        duration = time.time() - start_time
        stats["time"] = duration

        return stats, matrix_reconstructed


class App:

    def __init__(self):
        self.processor = MatrixProcessor(size=1000)
        self.root = tk.Tk()
        self.setup_window()

    def setup_window(self):
        self.root.title("Reto: Matrices e Imágenes")
        self.root.geometry("350x200")

        tk.Label(self.root, text="Procesamiento de Matrices", font=("Arial", 12, "bold")).pack(pady=10)
        tk.Label(self.root, text="Tamaño: 1000x1000", font=("Arial", 10)).pack()

        self.btn_run = tk.Button(
            self.root,
            text="EJECUTAR RETO",
            command=self.execute_challenge,
            bg="#3498db",
            fg="white",
            font=("Arial", 10, "bold"),
            padx=20,
            pady=10
        )
        self.btn_run.pack(pady=20)

    def execute_challenge(self):
        try:
            print(">>> Iniciando procesamiento nativo...")
            native_stats = self.processor.run_native_processing()

            print(">>> Iniciando procesamiento optimizado (Numpy)...")
            numpy_stats, final_matrix = self.processor.run_numpy_processing()

            self.print_comparison(native_stats, numpy_stats)
            self.display_image(final_matrix)

            messagebox.showinfo("Éxito", "Procesamiento completado. Revisa la consola y la imagen generada.")
        except Exception as e:
            messagebox.showerror("Error", f"Ocurrió un error: {e}")

    def print_comparison(self, s1, s2):
        print("\n" + "=" * 45)
        print(f"{'METRICA':<15} | {'NATIVO (s)':<12} | {'NUMPY (s)':<12}")
        print("-" * 45)

        metrics = [("Mínimo", "min"), ("Máximo", "max"), ("Media", "media"),
                   ("Desv. Est.", "std"), ("Tiempo Total", "time")]

        for label, key in metrics:
            print(f"{label:<15} | {s1[key]:<12.4f} | {s2[key]:<12.4f}")
        print("=" * 45 + "\n")

    def display_image(self, matrix):
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle("Comparativa de Mapas de Color (1000x1000)", fontsize=16)

        colormaps = ['magma', 'viridis', 'jet', 'gray']

        for i, ax in enumerate(axes.flat):
            cmap_name = colormaps[i]
            im = ax.imshow(matrix, cmap=cmap_name)
            ax.set_title(f"Mapa: {cmap_name}")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        output_path = os.path.join(self.processor.folder_imgs, "comparativa_colores.png")
        plt.savefig(output_path)
        print(f"SISTEMA: Comparativa guardada en: {output_path}")
        plt.show()

    def run(self):
        self.root.mainloop()


def main():
    app = App()
    app.run()


if __name__ == "__main__":
    main()
