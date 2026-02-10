import os
import random
import time
import cv2
import numpy as np


class ImageManager:

    def __init__(self, subfolder="images"):
        self.base_path = os.path.dirname(os.path.abspath(__file__))
        self.assets_path = os.path.join(self.base_path, subfolder)

        if not os.path.exists(self.assets_path):
            os.makedirs(self.assets_path)

    def load_grayscale(self, image_name):
        path = os.path.join(self.assets_path, image_name)
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"No se encontró la imagen en: {path}")
        return image

    def save_results(self, images_dict):
        for name, img in images_dict.items():
            save_path = os.path.join(self.assets_path, f"{name}.jpg")
            cv2.imwrite(save_path, img)
            print(f"ARCHIVO: Guardado {save_path}")


class FilterProcessor:

    @staticmethod
    def add_salt_and_pepper(image, prob=0.02):
        noisy = np.copy(image)
        height, width = image.shape
        for i in range(height):
            for j in range(width):
                rdn = random.random()
                if rdn < prob:
                    noisy[i][j] = 0  # Pimienta
                elif rdn > (1 - prob):
                    noisy[i][j] = 255  # Sal
        return noisy

    @staticmethod
    def manual_median_filter(image, kernel_size=3):
        output = np.zeros_like(image)
        offset = kernel_size // 2
        rows, cols = image.shape

        print("SISTEMA: Iniciando convolución manual (Mediana)...")
        for i in range(offset, rows - offset):
            for j in range(offset, cols - offset):
                window = image[i - offset: i + offset + 1, j - offset: j + offset + 1]
                output[i, j] = np.median(window)
        return output


class App:
    def __init__(self, target_image="original.jpg"):
        self.image_manager = ImageManager()
        self.processor = FilterProcessor()
        self.target_image = target_image

    def create_mosaic(self, images):
        top_row = np.hstack((images[0], images[1]))
        bottom_row = np.hstack((images[2], images[3]))
        grid = np.vstack((top_row, bottom_row))

        screen_res = 1280, 720
        scale = min(screen_res[0] / grid.shape[1], screen_res[1] / grid.shape[0])

        if scale < 1.0:
            new_size = (int(grid.shape[1] * scale), int(grid.shape[0] * scale))
            return cv2.resize(grid, new_size)
        return grid

    def run(self):
        try:
            img_gray = self.image_manager.load_grayscale(self.target_image)
            noisy_img = self.processor.add_salt_and_pepper(img_gray)

            start = time.time()
            manual_res = self.processor.manual_median_filter(noisy_img)
            t_manual = time.time() - start

            start = time.time()
            opencv_res = cv2.medianBlur(noisy_img, 3)
            t_opencv = time.time() - start

            results_dict = {
                "original-gray": img_gray,
                "ruido-salt-pepper": noisy_img,
                "filtro-manual": manual_res,
                "filtro-opencv": opencv_res
            }
            self.image_manager.save_results(results_dict)

            print("\n" + "=" * 40)
            print(f"{'MÉTODO':<15} | {'TIEMPO (s)':<15}")
            print("-" * 40)
            print(f"{'Manual':<15} | {t_manual:<15.4f}")
            print(f"{'OpenCV':<15} | {t_opencv:<15.4f}")
            print("=" * 40)

            mosaic = self.create_mosaic([img_gray, noisy_img, manual_res, opencv_res])
            cv2.imshow("Reto Convoluciones: Cuadricula 2x2", mosaic)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        except Exception as e:
            print(f"ERROR EN LA APLICACIÓN: {e}")


def main():
    app = App("original.jpg")
    app.run()


if __name__ == "__main__":
    main()