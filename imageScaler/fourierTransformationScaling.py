import cv2
import numpy as np
from matplotlib import pyplot as plt

def scale_image(image, scale_factor):
    # Perform Fourier Transform
    f_transform = np.fft.fft2(image)
    f_transform_shifted = np.fft.fftshift(f_transform)

    # Get image dimensions
    rows, cols = image.shape

    # Frequency domain coordinates
    u = np.fft.fftfreq(rows)
    v = np.fft.fftfreq(cols)

    # Meshgrid for frequency domain coordinates
    u, v = np.meshgrid(u, v)

    # Scaling operation in the frequency domain
    f_transform_scaled = f_transform_shifted * np.exp(-2j * np.pi * (u + v) * (1 - scale_factor) / 2)

    # Inverse Fourier Transform
    f_transform_scaled_shifted = np.fft.ifftshift(f_transform_scaled)
    image_scaled = np.abs(np.fft.ifft2(f_transform_scaled_shifted))

    # Clip to ensure pixel values are within the valid range
    image_scaled = np.clip(image_scaled, 0, 255)

    return np.uint8(image_scaled)


# Load an example image (replace 'path/to/your/image.jpg' with the actual path)
original_image = cv2.imread('path/to/your/image.jpg', cv2.IMREAD_GRAYSCALE)

# Choose a scale factor (adjust as needed)
scale_factor = 0.5  # For scaling down by half

# Scale the image
scaled_image = scale_image(original_image, scale_factor)

# Display the original and scaled images side by side
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(original_image, cmap='gray')
plt.title('Original Image')

plt.subplot(1, 2, 2)
plt.imshow(scaled_image, cmap='gray')
plt.title('Scaled Image')

plt.show()
