import requests
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import tkinter as tk
from tkinter import messagebox


class TrueRandomProcessor:
    """Clase para obtener números aleatorios reales de RANDOM.ORG."""

    def __init__(self):
        self.base_url = "https://www.random.org/integers/"
        self.folder_imgs = "images"
        if not os.path.exists(self.folder_imgs):
            os.makedirs(self.folder_imgs)

    def fetch_true_random_numbers(self, amount, min_val=0, max_val=255):
        """
        Método recursivo para obtener la cantidad exacta de números.
        RANDOM.ORG limita a 10,000 por petición.
        """
        limit = 10000

        # Caso base: Si pedimos 0 o menos
        if amount <= 0:
            return []

        # Determinar cuántos pedir en esta iteración
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
            # Convertir texto plano a lista de enteros
            numbers = [int(n) for n in response.text.split() if n.strip()]

            # Llamada recursiva para el resto de los números
            return numbers + self.fetch_true_random_numbers(amount - current_request_size)

        except Exception as e:
            print(f"ERROR: Fallo en la conexión: {e}")
            return []

    def generate_matrix(self, rows, cols):
        """Genera una matriz NxM usando la recursión de peticiones."""
        total_elements = rows * cols
        data = self.fetch_true_random_numbers(total_elements)

        if len(data) != total_elements:
            raise ValueError("No se obtuvieron suficientes datos de la API.")

        return np.array(data).reshape((rows, cols))


class App:
    """Clase principal de la interfaz."""

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
        """Muestra la matriz en diferentes colormaps."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f"Aleatoriedad Real (Atmosférica) - {self.n}x{self.m}", fontsize=14)

        # Usamos Gray para ver el 'ruido' puro y Magma para intensidad
        axes[0].imshow(matrix, cmap='gray')
        axes[0].set_title("Mapa: Gray (Ruido Puro)")

        im2 = axes[1].imshow(matrix, cmap='magma')
        axes[1].set_title("Mapa: Magma (Intensidad)")
        fig.colorbar(im2, ax=axes[1])

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    app = App(n=100, m=100)
    app.root.mainloop()