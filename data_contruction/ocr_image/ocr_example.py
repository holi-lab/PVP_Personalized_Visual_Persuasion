import easyocr
from PIL import Image
import numpy as np

def resize_image(image_path, target_size=(1024, 1024)):
    """Resize an image to the target size while maintaining aspect ratio."""
    with Image.open(image_path) as img:
        img.thumbnail(target_size, Image.LANCZOS)
        new_img = Image.new("RGB", target_size, (255, 255, 255))
        new_img.paste(img, ((target_size[0] - img.size[0]) // 2,
                            (target_size[1] - img.size[1]) // 2))
        return np.array(new_img)

def perform_ocr(image_path, language='en', use_gpu=False):
    """
    Perform OCR on a single image and return the text count and OCR results.
    
    Parameters:
    image_path (str): Path to the image file.
    language (str): The language of the text to be recognized. Default is 'en'.
    use_gpu (bool): Whether to use GPU for text recognition. Default is True.
    
    Returns:
    tuple: (text_count, ocr_results)
    """
    # Initialize EasyOCR Reader
    reader = easyocr.Reader([language], gpu=use_gpu)
    
    # Resize image
    resized_image = resize_image(image_path)
    
    # Perform OCR
    results = reader.readtext(resized_image)
    
    text_count = 0
    filtered_results = []
    
    for bbox, text, confidence in results:
        if confidence >= 0.95:
            text_count += len(text)
            filtered_results.append((bbox, text, confidence))
    
    return text_count, filtered_results

if __name__ == "__main__":
    # Example usage
    image_path = "./example_image.jpg"  # Replace with the path to your image
    
    text_count, ocr_results = perform_ocr(image_path)
    
    print(f"Total characters: {text_count}")
    print("OCR Results:")
    for bbox, text, confidence in ocr_results:
        print(f"Text: {text}, Confidence: {confidence:.2f}")
    
    # Set threshold to filter out text heavy images
    threshold = 10
    if text_count > threshold:
        print("Text heavy image detected!")
    else:
        print("Not a text heavy image.")