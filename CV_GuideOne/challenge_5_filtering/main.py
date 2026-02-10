import os
import random
import time

import cv2
import numpy as np


class ConvolutionChallenge:

    def __init__(self, image_name="original.jpg"):
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.assets_path = os.path.join(self.current_dir, "images")
        self.image_path = os.path.join(self.assets_path, image_name)
        self.image = None

        if not os.path.exists(self.assets_path):
            os.makedirs(self.assets_path)

    def load_image(self):
        self.image = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
        if self.image is None:
            raise FileNotFoundError(f"No se encontró la imagen en {self.image_path}")
        return self.image

    def add_salt_and_pepper(self, image, prob=0.05):
        noisy = np.copy(image)
        thres = 1 - prob
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                rdn = random.random()
                if rdn < prob:
                    noisy[i][j] = 0  # Pimienta
                elif rdn > thres:
                    noisy[i][j] = 255  # Sal
        return noisy

    def manual_median_filter(self, image, kernel_size=3):
        output = np.zeros_like(image)
        offset = kernel_size // 2
        rows, cols = image.shape

        print("SISTEMA: Procesando convolución manual...")

        for i in range(offset, rows - offset):
            for j in range(offset, cols - offset):
                window = image[i - offset: i + offset + 1, j - offset: j + offset + 1]
                output[i, j] = np.median(window)

        return output

    def save_images(self, images_dict):
        for name, img in images_dict.items():
            save_path = os.path.join(self.assets_path, f"{name}.jpg")
            cv2.imwrite(save_path, img)
            print(f"ARCHIVO: Guardado {save_path}")

    def run_comparison(self):
        img = self.load_image()

        noisy_img = self.add_salt_and_pepper(img, prob=0.02)

        start_manual = time.time()
        manual_denoised = self.manual_median_filter(noisy_img, kernel_size=3)
        time_manual = time.time() - start_manual

        start_cv2 = time.time()
        opencv_denoised = cv2.medianBlur(noisy_img, 3)
        time_cv2 = time.time() - start_cv2

        results = {
            "original-gray": img,
            "ruido-salt-pepper": noisy_img,
            "filtro-manual": manual_denoised,
            "filtro-opencv": opencv_denoised
        }

        print("\n" + "=" * 40)
        print(f"{'MÉTODO':<15} | {'TIEMPO (s)':<15}")
        print("-" * 40)
        print(f"{'Manual':<15} | {time_manual:<15.4f}")
        print(f"{'OpenCV':<15} | {time_cv2:<15.4f}")
        print("=" * 40)

        self.save_images(results)
        self.show_results_grid(img, noisy_img, manual_denoised, opencv_denoised)

    def show_results_grid(self, orig, noisy, manual, cv2_res):
        top_row = np.hstack((orig, noisy))
        bottom_row = np.hstack((manual, cv2_res))

        grid = np.vstack((top_row, bottom_row))

        screen_res = 1280, 720
        scale_width = screen_res[0] / grid.shape[1]
        scale_height = screen_res[1] / grid.shape[0]
        scale = min(scale_width, scale_height)

        if scale < 1.0:
            window_size = (int(grid.shape[1] * scale), int(grid.shape[0] * scale))
            grid_display = cv2.resize(grid, window_size)
        else:
            grid_display = grid

        window_name = "Mosaico: Original | Ruido | Manual | OpenCV"
        cv2.imshow(window_name, grid_display)

        print("\nSISTEMA: Mostrando cuadrícula. Presione cualquier tecla para cerrar.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    try:
        challenge = ConvolutionChallenge("original.jpg")
        challenge.run_comparison()
    except Exception as e:
        print(f"ERROR: {e}")


if __name__ == "__main__":
    main()
