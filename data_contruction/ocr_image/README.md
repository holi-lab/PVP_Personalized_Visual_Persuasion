# OCR Example Script

This Python script demonstrates how to perform Optical Character Recognition (OCR) on a single image using the EasyOCR library. It includes functionality to resize the image, perform OCR, and filter results based on confidence levels.

## Features

- Resize input images to a standard size while maintaining aspect ratio
- Perform OCR on images using EasyOCR
- Filter OCR results based on confidence levels
- Count total characters in recognized text
- Detect if an image is text-heavy based on a character count threshold

## Prerequisites

- Python 3.x
- PIL (Python Imaging Library)
- NumPy
- EasyOCR

## Installation

1. Ensure you have Python 3.x installed on your system.
2. Install the required libraries:

   ```
   pip install pillow numpy easyocr
   ```

## Usage

1. Place your image file in the same directory as the script or update the `image_path` variable with the correct path to your image.

2. Run the script:

   ```
   python ocr_example.py
   ```

3. The script will output:
   - Total character count
   - Recognized text with confidence levels
   - Whether the image is considered text-heavy based on the character count

## Customization

- You can modify the `target_size` in the `resize_image` function to change the resizing dimensions.
- Adjust the confidence threshold (currently set to 0.95) in the `perform_ocr` function to include or exclude results.
- Change the `threshold` value (currently set to 10) to adjust when an image is considered text-heavy.

## Note

- The script is currently set to use CPU for OCR. If you have a compatible GPU and want to use it, change `use_gpu=False` to `use_gpu=True` in the `perform_ocr` function call.