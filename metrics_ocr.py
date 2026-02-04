from pathlib import Path
import os
import json
from argparse import ArgumentParser

from metrics_ocr.run_ocr import run_ocr
from metrics_ocr.visualize_ocr import visualize_folder
from metrics_ocr.get_ocr_results import evaluate_cer


if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    args = parser.parse_args()

    for model_path in args.model_paths:
        print(f"OCR Evaluation for model: {model_path}")

        test_dir = Path(model_path) / "test"
        ocr_output_dir = Path(model_path) / "test_ocr_output"
        ocr_output_file = ocr_output_dir / "ocr_results.json"

        for method in os.listdir(test_dir):
            print("Method:", method)
            method_dir = test_dir / method
            gt_dir = method_dir / "gt"
            renders_dir = method_dir / "renders"
            masks_dir = method_dir / "masks"

            gt_output_folder = ocr_output_dir / "gt"
            gt_ocr_jsons = gt_output_folder / "ocr_jsons"
            gt_ocr_visualizations = gt_output_folder / "visualizations"
            run_ocr(str(gt_dir), str(masks_dir), str(gt_ocr_jsons))
            visualize_folder(str(gt_dir), str(gt_ocr_jsons), str(gt_ocr_visualizations))

            renders_output_folder = ocr_output_dir / method
            renders_ocr_jsons = renders_output_folder / "ocr_jsons"
            renders_ocr_visualizations = renders_output_folder / "visualizations"
            run_ocr(str(renders_dir), str(masks_dir), str(renders_ocr_jsons))
            visualize_folder(str(renders_dir), str(renders_ocr_jsons), str(renders_ocr_visualizations))

            results = evaluate_cer(str(gt_ocr_jsons), str(renders_ocr_jsons))

            existing_results = {}
            if ocr_output_file.exists():
                with open(ocr_output_file, 'r') as f:
                    existing_results = json.load(f)
            existing_results[method] = results
            with open(ocr_output_file, 'w') as f:
                json.dump(existing_results, f, indent=2)

        print("OCR Evaluation complete for model:", model_path)
    
            

        

