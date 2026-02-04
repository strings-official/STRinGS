import os
import json
import cv2
import numpy as np

def draw_polygons(image_path, regions, output_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not read image: {image_path}")
        return
    
    for region in regions:
        polygon = np.array(region['bounds'], dtype=np.int32)
        
        cv2.polylines(img, [polygon], isClosed=True, color=(0, 255, 0), thickness=2)
        
        text = region['text']
        x, y = polygon[0][0], polygon[0][1] - 10
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    cv2.imwrite(output_path, img)

def visualize_folder(image_folder, json_folder, output_vis_folder):
    """Visualize OCR results for all images in a folder."""
    if not os.path.exists(output_vis_folder):
        os.makedirs(output_vis_folder, exist_ok=True)
    else:
        print(f"Visualization folder {output_vis_folder} already exists. Skipping...")
        return
    
    for json_file in os.listdir(json_folder):
        if not json_file.endswith('.json'):
            continue
            
        base_name = os.path.splitext(json_file)[0]
        img_name = base_name + '.png'  
        
        img_path = os.path.join(image_folder, img_name)
        if not os.path.exists(img_path):
            continue
            
        with open(os.path.join(json_folder, json_file), 'r') as f:
            regions = json.load(f)
        
        output_path = os.path.join(output_vis_folder, f"{base_name}_vis.png")
        draw_polygons(img_path, regions, output_path)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualize OCR results on images.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model directory.")
    parser.add_argument("--iteration", type=int, default=0, help="Iteration number to process.")

    args = parser.parse_args()
    model_path = args.model_path
    gt_folder = os.path.join(model_path, "eval_test", "gt")
    renders_folder = os.path.join(model_path, "eval_test", f"renders_{args.iteration}")
    output_folder = os.path.join(model_path, "eval_test", "ocr_output")

    visualize_folder(gt_folder, os.path.join(output_folder, "gt", "prediction_jsons"), os.path.join(output_folder, "gt", "visualizations"))
    visualize_folder(renders_folder, os.path.join(output_folder, f"renders_{args.iteration}", "prediction_jsons"), os.path.join(output_folder, f"renders_{args.iteration}", "visualizations"))

    print("Visualizations created")
