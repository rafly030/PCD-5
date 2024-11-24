import cv2
import numpy as np
import matplotlib.pyplot as plt

# Fungsi untuk menerapkan filter pada citra
def apply_filter(image, kernel):
    return cv2.filter2D(image, -1, kernel)

# Membaca citra grayscale dan berwarna
image_gray = cv2.imread('Dinosaurus.jpg', cv2.IMREAD_GRAYSCALE)
image_color = cv2.imread('Dinosauruss.jpg')
image_color = cv2.cvtColor(image_color, cv2.COLOR_BGR2RGB)  # Konversi ke RGB untuk plt.imshow

# Filter low-pass (blur)
kernel_low_pass = np.ones((3, 3), np.float32) / 9  # Kernel rata-rata 3x3
gray_low_pass = apply_filter(image_gray, kernel_low_pass)
color_low_pass = cv2.filter2D(image_color, -1, kernel_low_pass)

# Filter high-pass (deteksi tepi)
kernel_high_pass = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])  # Kernel high-pass 3x3
gray_high_pass = apply_filter(image_gray, kernel_high_pass)
color_high_pass = np.zeros_like(image_color)
for i in range(3):  # Terapkan ke setiap saluran warna
    color_high_pass[:, :, i] = apply_filter(image_color[:, :, i], kernel_high_pass)

# Filter high-boost (detail enhancement)
alpha = 1.5  # Faktor penguatan
kernel_high_boost = kernel_low_pass * (1 - alpha) + np.eye(3) * alpha
gray_high_boost = apply_filter(image_gray, kernel_high_boost)
color_high_boost = np.zeros_like(image_color)
for i in range(3):  # Terapkan ke setiap saluran warna
    color_high_boost[:, :, i] = apply_filter(image_color[:, :, i], kernel_high_boost)

# Visualisasi hasil
plt.figure(figsize=(15, 10))

# Citra grayscale
plt.subplot(3, 4, 1), plt.imshow(image_gray, cmap='gray'), plt.title('Grayscale Original')
plt.subplot(3, 4, 2), plt.imshow(gray_low_pass, cmap='gray'), plt.title('Grayscale Low-pass')
plt.subplot(3, 4, 3), plt.imshow(gray_high_pass, cmap='gray'), plt.title('Grayscale High-pass')
plt.subplot(3, 4, 4), plt.imshow(gray_high_boost, cmap='gray'), plt.title('Grayscale High-boost')

# Citra berwarna
plt.subplot(3, 4, 5), plt.imshow(image_color), plt.title('Color Original')
plt.subplot(3, 4, 6), plt.imshow(color_low_pass), plt.title('Color Low-pass')
plt.subplot(3, 4, 7), plt.imshow(color_high_pass), plt.title('Color High-pass')
plt.subplot(3, 4, 8), plt.imshow(color_high_boost), plt.title('Color High-boost')

# Memberikan jarak antar subplot
plt.subplots_adjust(wspace=1, hspace=2)  # Atur jarak antar gambar

plt.tight_layout()
plt.show()
