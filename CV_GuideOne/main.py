import tkinter as tk
from tkinter import ttk, messagebox


class MainDashboard:

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Computer Vision - Guía 1")
        self.root.geometry("450x350")
        self.root.configure(bg="#f0f2f5")

        self.setup_styles()
        self.create_widgets()

    def setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')

        style.configure("Challenge.TButton",
                        font=("Helvetica", 11, "bold"),
                        padding=10,
                        background="#ffffff")

    def create_widgets(self):
        """Crea los componentes visuales del dashboard."""
        # Título principal
        header = tk.Label(
            self.root,
            text="Panel de Control de Retos",
            font=("Helvetica", 16, "bold"),
            bg="#f0f2f5",
            fg="#1a73e8"
        )
        header.pack(pady=20)

        btn_container = tk.Frame(self.root, bg="#f0f2f5")
        btn_container.pack(expand=True, fill="both", padx=50)

        btn_one = ttk.Button(
            btn_container,
            text="1. Matrices a Imágenes",
            style="Challenge.TButton",
            command=self.launch_challenge_one
        )
        btn_one.pack(fill="x", pady=10)

        btn_three = ttk.Button(
            btn_container,
            text="3. Colores",
            style="Challenge.TButton",
            command=self.launch_challenge_three
        )
        btn_three.pack(fill="x", pady=10)

        btn_four = ttk.Button(
            btn_container,
            text="4. Seguimiento por Color",
            style="Challenge.TButton",
            command=self.launch_challenge_four
        )
        btn_four.pack(fill="x", pady=10)

        btn_five = ttk.Button(
            btn_container,
            text="5. Filtro de Ruido",
            style="Challenge.TButton",
            command=self.launch_challenge_five
        )
        btn_five.pack(fill="x", pady=10)

        btn_exit = tk.Button(
            self.root,
            text="SALIR",
            command=self.root.quit,
            bg="#d93025",
            fg="white",
            font=("Helvetica", 10, "bold"),
            relief="flat"
        )
        btn_exit.pack(side="bottom", fill="x", pady=20, padx=50)

    def launch_challenge_one(self):
        try:
            print("SISTEMA: Iniciando Reto 1 - Matrices...")
            from challenge_1_matrices.main import main as start_matrix_challenge
            start_matrix_challenge()
        except ImportError:
            messagebox.showerror("Error de Importación", "No se encontró el archivo: challenge_1_matrices/main.py")
        except Exception as e:
            messagebox.showerror("Error", f"Ocurrió un error al lanzar el reto: {e}")

    def launch_challenge_three(self):
        try:
            print("SISTEMA: Iniciando Reto 3 - Colores...")
            from challenge_3_colors.main import main as start_color_challenge
            start_color_challenge()
        except ImportError:
            messagebox.showerror("Error de Importación", "No se encontró el archivo: challenge_3_colors/main.py")
        except Exception as e:
            messagebox.showerror("Error", f"Ocurrió un error al lanzar el reto: {e}")

    def launch_challenge_four(self):
        try:
            print("SISTEMA: Iniciando Reto 4 - Seguimiento por Color...")
            from challenge_4_object_tracking.main import main as start_tracking_challenge
            start_tracking_challenge()
        except ImportError:
            messagebox.showerror("Error de Importación",
                                 "No se encontró el archivo: challenge_4_object_tracking/main.py")
        except Exception as e:
            messagebox.showerror("Error", f"Ocurrió un error al lanzar el reto: {e}")

    def launch_challenge_five(self):
        try:
            print("SISTEMA: Iniciando Reto 5 - Filtro de Ruido...")
            from challenge_5_filtering.main import main as start_filtering_challenge
            start_filtering_challenge()
        except ImportError:
            messagebox.showerror("Error de Importación",
                                 "No se encontró el archivo: challenge_5_filtering/main.py")
        except Exception as e:
            messagebox.showerror("Error", f"Ocurrió un error al lanzar el reto: {e}")

    def run(self):
        self.root.mainloop()


def main():
    print("SISTEMA: Cargando Dashboard Principal...")
    app = MainDashboard()
    app.run()


if __name__ == "__main__":
    main()