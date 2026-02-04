import os
import json
from google.cloud import vision
from shapely.geometry import Polygon
from tqdm import tqdm
import cv2
import numpy as np

client = vision.ImageAnnotatorClient()

def polygon_from_vertices(vertices):
    return Polygon([(vertex.x, vertex.y) for vertex in vertices])

def multiply_image_with_mask(image_path, mask_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Could not load mask from {mask_path}")

    binary_mask = (mask > 127).astype(np.uint8)
    binary_mask_3c = cv2.merge([binary_mask]*3) 

    masked_image = image * binary_mask_3c

    success, buffer = cv2.imencode('.png', masked_image)
    if not success:
        raise ValueError("Could not encode image to binary format.")

    content = buffer.tobytes()
    return content

def ocr_api(image_path, mask_path):
    """Run OCR and return detected text regions with polygon bounds."""
    try:
        content = multiply_image_with_mask(image_path, mask_path)
        image = vision.Image(content=content)
        response = client.text_detection(image=image)
        
        if response.error.message:
            print(f"API Error: {response.error.message}")
            return None
        
        texts = response.text_annotations
        if not texts:
            return []
        
        regions = []
        for text in texts[1:]:  # Skip first annotation (aggregated text)
            try:
                poly = polygon_from_vertices(text.bounding_poly.vertices)
                if not poly.is_valid:
                    poly = poly.convex_hull
                regions.append({
                    'text': text.description,
                    'polygon': [(v.x, v.y) for v in text.bounding_poly.vertices],
                    'bounds': [(v.x, v.y) for v in text.bounding_poly.vertices]
                })
            except Exception as e:
                print(f"Error processing polygon for text '{text.description}': {str(e)}")
                continue
        return regions
    except Exception as e:
        print(f"OCR failed for {image_path}: {str(e)}")
        return None

def run_ocr(input_folder, masks_folder, output_json_dir):
    """Process all images in a folder and save OCR results as JSON."""
    if os.path.exists(output_json_dir):
        print(f"Output directory {output_json_dir} already exists. Skipping...")
        return
    
    os.makedirs(output_json_dir, exist_ok=True)
    
    results = {}
    for img_name in tqdm(os.listdir(input_folder)):
        if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
            
        img_path = os.path.join(input_folder, img_name)
        mask_path = os.path.join(masks_folder, img_name)
        regions = ocr_api(img_path, mask_path)
        
        if regions is not None:
            output_path = os.path.join(output_json_dir, f"{os.path.splitext(img_name)[0]}.json")
            with open(output_path, 'w') as f:
                json.dump(regions, f, indent=2)
            results[img_name] = regions
    
    return results
    

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run OCR on images and save results as JSON.")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the model directory.")
    parser.add_argument('--iteration', type=int, required=True, help="Iteration number to process.")

    args = parser.parse_args()
    model_path = args.model_path
    gt_folder = os.path.join(model_path, "eval_test", "gt")
    renders_folder = os.path.join(model_path, "eval_test", f"renders_{args.iteration}")
    output_folder = os.path.join(model_path, "eval_test", "ocr_output")
    masks_folder = os.path.join(model_path, "eval_test", "masks")

    run_ocr(gt_folder, masks_folder, os.path.join(output_folder, "gt", "prediction_jsons"))
    run_ocr(renders_folder, masks_folder, os.path.join(output_folder, f"renders_{args.iteration}", "prediction_jsons"))

    print("Running OCR complete")
