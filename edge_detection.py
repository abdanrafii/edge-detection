import cv2
import numpy as np
from scipy import ndimage

def read_image(file):
    """Read uploaded image (BytesIO) into OpenCV format"""
    file_bytes = np.frombuffer(file.read(), np.uint8)
    bgr_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    return rgb_image

def to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def apply_sobel(image):
    gray = to_grayscale(image)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    sobelx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(sobelx, sobely)

    return cv2.convertScaleAbs(magnitude)

def apply_prewitt(image):
    gray = to_grayscale(image)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    kernel_x = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    kernel_y = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])

    prewitt_x = cv2.filter2D(blur, cv2.CV_64F, kernel_x)
    prewitt_y = cv2.filter2D(blur, cv2.CV_64F, kernel_y)

    magnitude = cv2.magnitude(prewitt_x, prewitt_y)
    return cv2.convertScaleAbs(magnitude)

def apply_roberts(image):
    gray = to_grayscale(image)

    gx = np.array([[1, 0], [0, -1]], dtype=np.float32)
    gy = np.array([[0, 1], [-1, 0]], dtype=np.float32)

    grad_x = cv2.filter2D(gray, cv2.CV_64F, gx)
    grad_y = cv2.filter2D(gray, cv2.CV_64F, gy)

    magnitude = cv2.magnitude(grad_x, grad_y)
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    edges = (magnitude > 30).astype(np.uint8) * 255
    return edges

def apply_compass(image, direction="All Directions"):
    gray = to_grayscale(image)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    compass_kernels = {
        "North": np.array([[1, 1, 1], [1, -2, 1], [-1, -1, -1]]),
        "Northeast": np.array([[1, 1, -1], [1, -2, 1], [-1, 1, 1]]),
        "East": np.array([[-1, 1, 1], [-1, -2, 1], [-1, 1, 1]]),
        "Southeast": np.array([[-1, 1, 1], [-1, -2, 1], [1, 1, -1]]),
        "South": np.array([[-1, -1, -1], [1, -2, 1], [1, 1, 1]]),
        "Southwest": np.array([[1, 1, -1], [1, -2, 1], [1, -1, 1]]),
        "West": np.array([[1, 1, -1], [1, -2, -1], [1, 1, -1]]),
        "Northwest": np.array([[1, -1, -1], [1, -2, 1], [1, 1, 1]])
    }

    if direction == "All Directions":
        responses = [cv2.filter2D(blurred, cv2.CV_64F, k) for k in compass_kernels.values()]
        edge_map = np.max(np.abs(responses), axis=0)
    else:
        kernel = compass_kernels[direction]
        edge_map = cv2.filter2D(blurred, cv2.CV_64F, kernel)
        edge_map = np.abs(edge_map)

    edges = cv2.convertScaleAbs(edge_map)
    return edges

def apply_log(image, sigma=1.0, noise_intensity=0.0):
    if noise_intensity > 0:
        noise = np.random.normal(0, noise_intensity, image.shape)
        image = image + noise
        image = np.clip(image, 0, 255).astype(np.uint8)

    if sigma > 0:
        blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    else:
        blurred = image

    log = cv2.Laplacian(blurred, cv2.CV_64F)

    log = np.abs(log)
    log = (log / log.max() * 255).astype(np.uint8)

    return log