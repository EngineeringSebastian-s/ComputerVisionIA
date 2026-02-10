import os
import tkinter as tk
from tkinter import ttk, messagebox
import openpyxl
import pandas as pd


class DocumentViewer:
    def __init__(self):
        self.base_path = os.path.dirname(os.path.abspath(__file__))
        self.excel_path = os.path.join(self.base_path, "KERNELS.xlsx")
        self.pdf_path = os.path.join(self.base_path, "RETO_2.pdf")

    def open_pdf_system(self):
        try:
            if os.path.exists(self.pdf_path):
                # 'start' es para Windows. En Mac usaría 'open', en Linux 'xdg-open'
                os.startfile(self.pdf_path)
            else:
                messagebox.showerror("Error", "No se encontró el archivo RETO_2.pdf")
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo abrir el PDF: {e}")

    def show_excel_table(self):
        try:
            if not os.path.exists(self.excel_path):
                messagebox.showerror("Error", "No se encontró KERNELS.xlsx")
                return

            df = pd.read_excel(self.excel_path)

            top = tk.Toplevel()
            top.title("Vista Previa: KERNELS.xlsx")
            top.geometry("800x400")

            tree = ttk.Treeview(top)

            tree["columns"] = list(df.columns)
            tree["show"] = "headings"

            for col in df.columns:
                tree.heading(col, text=col)
                tree.column(col, width=100)

            for _, row in df.iterrows():
                tree.insert("", "end", values=list(row))

            scroll_y = ttk.Scrollbar(top, orient="vertical", command=tree.yview)
            scroll_x = ttk.Scrollbar(top, orient="horizontal", command=tree.xview)
            tree.configure(yscrollcommand=scroll_y.set, xscrollcommand=scroll_x.set)

            scroll_y.pack(side="right", fill="y")
            scroll_x.pack(side="bottom", fill="x")
            tree.pack(expand=True, fill="both")

        except Exception as e:
            messagebox.showerror("Error", f"Fallo al leer Excel: {e}")


class App:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Visualizador de Documentos - Reto 2")
        self.root.geometry("400x250")
        self.viewer = DocumentViewer()
        self.setup_ui()

    def setup_ui(self):
        tk.Label(self.root, text="Documentos del Proyecto", font=("Arial", 12, "bold")).pack(pady=20)

        tk.Button(self.root, text="Ver Tabla de KERNELS (Excel)",
                  command=self.viewer.show_excel_table, bg="#217346", fg="white", width=25).pack(pady=10)

        tk.Button(self.root, text="Abrir Guía del Reto (PDF)",
                  command=self.viewer.open_pdf_system, bg="#f44336", fg="white", width=25).pack(pady=10)

    def run(self):
        self.root.mainloop()


def main():
    app = App()
    app.run()


if __name__ == "__main__":
    main()
