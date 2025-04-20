# Visi-Komputer-
Tugas
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import erosion
from skimage.util import invert
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from google.colab import files

# Fungsi transformasi hit-or-miss
def hit_or_miss(image, se_foreground, se_background):
    image_complement = invert(image)
    eroded_foreground = erosion(image, se_foreground)
    eroded_background = erosion(image_complement, se_background)
    hitmiss_result = eroded_foreground & eroded_background
    return hitmiss_result

# Upload gambar
uploaded = files.upload()

# Membaca gambar pertama yang diupload
for filename in uploaded:
    img = imread(filename)
    break  # hanya ambil satu

# Konversi ke grayscale dan biner
if img.ndim == 3:
    img_gray = rgb2gray(img)
else:
    img_gray = img / 255.0

# Binarisasi menggunakan threshold otomatis
thresh = threshold_otsu(img_gray)
binary = img_gray < thresh  # hitam sebagai foreground

# Structuring element untuk foreground dan background
se_foreground = np.array([
    [0, 1, 0],
    [0, 1, 0],
    [0, 0, 0]
], dtype=bool)

se_background = np.array([
    [1, 0, 1],
    [1, 0, 1],
    [1, 1, 1]
], dtype=bool)

# Proses Hit-or-Miss
result = hit_or_miss(binary, se_foreground, se_background)

# Visualisasi hasil
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs[0].imshow(binary, cmap='gray')
axs[0].set_title("Binarized Image")

axs[1].imshow(result, cmap='gray')
axs[1].set_title("Hit-or-Miss Result")

axs[2].imshow(invert(binary), cmap='gray')
axs[2].set_title("Complement Image")

for ax in axs:
    ax.axis('off')

plt.tight_layout()
plt.show()
