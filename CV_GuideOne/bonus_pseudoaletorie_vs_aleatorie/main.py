import requests
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import tkinter as tk
from tkinter import messagebox


class TrueRandomProcessor:

    def __init__(self):
        self.base_url = "https://www.random.org/integers/"
        self.folder_imgs = "images"
        if not os.path.exists(self.folder_imgs):
            os.makedirs(self.folder_imgs)

    def fetch_true_random_numbers(self, amount, min_val=0, max_val=255):
        limit = 10000
        if amount <= 0:
            return []

        current_request_size = min(amount, limit)
        params = {
            'num': current_request_size,
            'min': min_val,
            'max': max_val,
            'col': 1,
            'base': 10,
            'format': 'plain',
            'rnd': 'new'
        }

        print(f"SISTEMA: Solicitando {current_request_size} números a RANDOM.ORG...")

        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            numbers = [int(n) for n in response.text.split() if n.strip()]
            return numbers + self.fetch_true_random_numbers(amount - current_request_size)
        except Exception as e:
            print(f"ERROR: Fallo en la conexión: {e}")
            return []

    def generate_matrix(self, rows, cols):
        total_elements = rows * cols
        data = self.fetch_true_random_numbers(total_elements)

        if len(data) != total_elements:
            raise ValueError("No se obtuvieron suficientes datos de la API.")

        return np.array(data).reshape((rows, cols))


class App:

    def __init__(self, n=100, m=100):
        self.n = n
        self.m = m
        self.processor = TrueRandomProcessor()
        self.root = tk.Tk()
        self.setup_gui()

    def setup_gui(self):
        self.root.title("Análisis de Aleatoriedad Atmosférica")
        self.root.geometry("400x250")

        tk.Label(self.root, text="RANDOM.ORG Visualizer", font=("Arial", 12, "bold")).pack(pady=10)
        tk.Label(self.root, text=f"Matriz objetivo: {self.n}x{self.m}").pack()

        self.btn_run = tk.Button(
            self.root, text="GENERAR DESDE RUIDO ATMOSFÉRICO",
            command=self.execute, bg="#e67e22", fg="white", font=("Arial", 10, "bold")
        )
        self.btn_run.pack(pady=20)

    def execute(self):
        try:
            start_time = time.time()
            matrix = self.processor.generate_matrix(self.n, self.m)
            duration = time.time() - start_time

            print(f"SISTEMA: Matriz generada en {duration:.2f}s")
            self.plot_comparison(matrix)

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def plot_comparison(self, matrix):
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f"Comparativa de Aleatoriedad Real - {self.n}x{self.m}", fontsize=16)

        colormaps = ['magma', 'viridis', 'jet', 'gray']

        for i, ax in enumerate(axes.flat):
            cmap_name = colormaps[i]
            im = ax.imshow(matrix, cmap=cmap_name)
            ax.set_title(f"Mapa: {cmap_name}")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        output_path = os.path.join(self.processor.folder_imgs, f"random_org_{self.n}x{self.m}.png")
        plt.savefig(output_path)

        print(f"SISTEMA: Imagen guardada en {output_path}")
        plt.show()


if __name__ == "__main__":
    app = App(n=100, m=100)
    app.root.mainloop()