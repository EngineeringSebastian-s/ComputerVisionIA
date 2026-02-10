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
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.folder_data = os.path.join(self.current_dir, "data")
        self.folder_imgs = os.path.join(self.current_dir, "images")
        self.file_path = os.path.join(self.folder_data, "matrix_data.csv")
        self.matrix = []

        for folder in [self.folder_data, self.folder_imgs]:
            if not os.path.exists(folder):
                os.makedirs(folder)

    def run_native_processing(self):
        t_start_creation = time.time()
        self.matrix = [[random.randint(0, 255) for _ in range(self.size)] for _ in range(self.size)]
        t_creation = time.time() - t_start_creation

        t_start_stats = time.time()
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
        t_stats = time.time() - t_start_stats

        t_start_save = time.time()
        flat_vector = [val for row in self.matrix for val in row]
        with open(self.file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(flat_vector)
        t_save = time.time() - t_start_save

        return {
            "min": min_val, "max": max_val, "media": mean_val, "std": std_dev,
            "t_creation": t_creation,
            "t_stats": t_stats,
            "t_save": t_save,
            "t_total": t_creation + t_stats + t_save
        }

    def run_numpy_processing(self):
        t_start_load = time.time()
        data_np = np.genfromtxt(self.file_path, delimiter=',')
        t_load = time.time() - t_start_load

        t_start_stats = time.time()
        stats = {
            "min": np.min(data_np),
            "max": np.max(data_np),
            "media": np.mean(data_np),
            "std": np.std(data_np)
        }
        t_stats = time.time() - t_start_stats

        matrix_reconstructed = data_np.reshape((self.size, self.size))

        stats["t_creation"] = t_load
        stats["t_stats"] = t_stats
        stats["t_save"] = 0
        stats["t_total"] = t_load + t_stats

        return stats, matrix_reconstructed


class App:

    def __init__(self):
        self.processor = MatrixProcessor(size=1000)
        self.root = tk.Tk()
        self.setup_window()

    def setup_window(self):
        self.root.title("Reto: Matrices e Imágenes")
        self.root.geometry("400x250")

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
            print(">>> Ejecutando Fase Nativa...")
            n_stats = self.processor.run_native_processing()

            print(">>> Ejecutando Fase NumPy...")
            np_stats, final_matrix = self.processor.run_numpy_processing()

            self.print_comparison(n_stats, np_stats)
            self.display_image(final_matrix)

            messagebox.showinfo("Proceso Terminado", "Los tiempos han sido segmentados en consola.")
        except Exception as e:
            messagebox.showerror("Error", f"Fallo en la ejecución: {e}")

    def print_comparison(self, s1, s2):
        print("\n" + "=" * 65)
        print(f"{'MÉTRICA / TIEMPO':<25} | {'NATIVO (s)':<15} | {'NUMPY (s)':<15}")
        print("-" * 65)

        print(f"{'Mínimo':<25} | {s1['min']:<15.4f} | {s2['min']:<15.4f}")
        print(f"{'Máximo':<25} | {s1['max']:<15.4f} | {s2['max']:<15.4f}")
        print(f"{'Media':<25} | {s1['media']:<15.4f} | {s2['media']:<15.4f}")
        print(f"{'Desv. Estándar':<25} | {s1['std']:<15.4f} | {s2['std']:<15.4f}")

        print("-" * 65)
        print(f"{'Creación/Carga':<25} | {s1['t_creation']:<15.4f} | {s2['t_creation']:<15.4f}")
        print(f"{'Cálculo Estadístico':<25} | {s1['t_stats']:<15.4f} | {s2['t_stats']:<15.4f}")
        print(f"{'Guardado (I/O)':<25} | {s1['t_save']:<15.4f} | {'N/A':<15}")

        print("-" * 65)
        print(f"{'TOTAL GENERAL':<25} | {s1['t_total']:<15.4f} | {s2['t_total']:<15.4f}")
        print("=" * 65 + "\n")

    def display_image(self, matrix):
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        fig.suptitle("Análisis Visual de la Matriz Generada", fontsize=14)
        colormaps = ['magma', 'viridis', 'jet', 'gray']

        for i, ax in enumerate(axes.flat):
            im = ax.imshow(matrix, cmap=colormaps[i])
            ax.set_title(f"Cmap: {colormaps[i]}")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    def run(self):
        self.root.mainloop()

def main():
    app = App()
    app.run()

if __name__ == "__main__":
    main()
