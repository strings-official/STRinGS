import os
import json
from Levenshtein import distance as edit_distance
from shapely.geometry import Polygon
import numpy as np
from collections import deque

def build_iou_graph(gt_regions, render_regions, iou_threshold=0.1):
    """Build bipartite graph between GT and render regions based on IoU."""
    graph = {}
    for i in range(len(gt_regions)):
        graph[f'gt_{i}'] = set()
    for j in range(len(render_regions)):
        graph[f'render_{j}'] = set()

    for i, gt in enumerate(gt_regions):
        poly_gt = Polygon(gt['polygon'])
        for j, render in enumerate(render_regions):
            poly_render = Polygon(render['polygon'])
            iou = compute_polygon_iou(gt['polygon'], render['polygon'])
            if iou >= iou_threshold:
                graph[f'gt_{i}'].add(f'render_{j}')
                graph[f'render_{j}'].add(f'gt_{i}')
    
    return graph

def find_connected_components(graph):
    """Find connected components in the IoU graph."""
    visited = set()
    components = []

    for node in graph.keys():
        if node not in visited:
            queue = deque([node])
            visited.add(node)
            component = set()

            while queue:
                n = queue.popleft()
                component.add(n)
                for neighbor in graph[n]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            
            components.append(component)
    
    return components


def calculate_metrics(gt_regions, render_regions, iou_threshold=0.1):
    """Calculate CER with region grouping for 1-N and N-1 matching."""
    graph = build_iou_graph(gt_regions, render_regions, iou_threshold)
    components = find_connected_components(graph)

    total_chars_gt = sum(len(r['text']) for r in gt_regions)

    char_errors = 0

    matched_gt = set()
    matched_render = set()

    matches = []

    for comp in components:
        gt_indices = [int(n.split('_')[1]) for n in comp if n.startswith('gt_')]
        render_indices = [int(n.split('_')[1]) for n in comp if n.startswith('render_')]

        if not gt_indices:
            continue  # purely false positives

        matched_gt.update(gt_indices)
        matched_render.update(render_indices)

        # Try all sorting combinations
        sort_options = ['x', 'y']
        min_char_error = float('inf')
        best_match = None

        for gt_sort in sort_options:
            for rd_sort in sort_options:
                # Sort GT
                gt_texts = [gt_regions[i]['text'].lower() for i in sort_by_axis(gt_indices, gt_regions, axis=gt_sort)]
                gt_text = ''.join(gt_texts).replace(' ', '')

                # Sort Render
                if render_indices:
                    rd_texts = [render_regions[j]['text'].lower() for j in sort_by_axis(render_indices, render_regions, axis=rd_sort)]
                    rd_text = ''.join(rd_texts).replace(' ', '')
                else:
                    rd_text = ''

                this_char_error = edit_distance(gt_text, rd_text)

                if this_char_error < min_char_error:
                    min_char_error = this_char_error
                    best_match = (gt_text, rd_text)

        matches.append(best_match)
        char_errors += min_char_error


    # Handle unmatched GT (missed regions)
    for i in range(len(gt_regions)):
        if i not in matched_gt:
            text = gt_regions[i]['text'].lower()
            char_errors += len(text)

    return {
        'cer': char_errors / total_chars_gt if total_chars_gt > 0 else 0,
        'num_render': len(render_regions),
        'num_matched': len(matched_render),
        'matches': matches
    }

def sort_by_axis(indices, regions, axis='x'):
    """Sort indices based on dominant axis."""
    if axis == 'x':
        key_fn = lambda i: np.mean([p[0] for p in regions[i]['polygon']])
    else:
        key_fn = lambda i: np.mean([p[1] for p in regions[i]['polygon']])
    return sorted(indices, key=key_fn)

def load_ocr_results(json_dir):
    """Load all OCR JSON results from a directory."""
    results = {}
    for json_file in os.listdir(json_dir):
        if not json_file.endswith('.json'):
            continue
        with open(os.path.join(json_dir, json_file), 'r') as f:
            img_name = json_file.replace('.json', '.png')  
            results[img_name] = json.load(f)
    return results

def compute_polygon_iou(poly1, poly2):
    """Compute Intersection over Union between two polygons."""
    poly1 = Polygon(poly1) if not isinstance(poly1, Polygon) else poly1
    poly2 = Polygon(poly2) if not isinstance(poly2, Polygon) else poly2
    
    if not poly1.is_valid or not poly2.is_valid:
        return 0.0
    
    intersection = poly1.intersection(poly2).area
    union = poly1.union(poly2).area
    
    return intersection / union if union > 0 else 0.0

def evaluate_cer(gt_json_dir, render_json_dir):
    """Evaluate render folder against GT OCR results."""
    gt_results = load_ocr_results(gt_json_dir)
    render_results = load_ocr_results(render_json_dir)

    folder_metrics = {
        'total_char_errors': 0,
        'total_chars_gt': 0,
        'images_processed': 0,
        'images_ignored': 0,
        'per_image': {}
    }
    
    for img_name, gt_regions in gt_results.items():
        if img_name not in render_results:
            continue
            
        if not gt_regions:
            folder_metrics['images_ignored'] += 1
            continue
            
        render_regions = render_results[img_name]
        metrics = calculate_metrics(gt_regions, render_regions)

        total_chars_gt = sum(len(r['text']) for r in gt_regions)
        total_words_gt = sum(len(r['text'].split()) for r in gt_regions)

        folder_metrics['total_char_errors'] += metrics['cer'] * total_chars_gt
        folder_metrics['total_chars_gt'] += total_chars_gt
        folder_metrics['images_processed'] += 1
        
        folder_metrics['per_image'][img_name] = {
            'cer': metrics['cer'],
            'num_gt_chars': total_chars_gt,
            'num_render': metrics['num_render'],
            'num_matched': metrics['num_matched'],
            'matches': metrics['matches']
        }
    
    overall_cer = (
        folder_metrics['total_char_errors'] / folder_metrics['total_chars_gt']
        if folder_metrics['total_chars_gt'] > 0 else 0
    )
    
    return {
        'overall_cer': overall_cer,
        'images_processed': folder_metrics['images_processed'],
        'images_ignored': folder_metrics['images_ignored'],
        'per_image': folder_metrics['per_image']
    }



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate OCR results from GT and render folders.")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the model directory.")
    parser.add_argument('--iteration', type=int, required=True, help="Iteration number to process.")
    parser.add_argument("--wandb_flag", action='store_true', help="Flag to indicate if results should be logged to WandB.")

    args = parser.parse_args()
    model_path = args.model_path

    gt_json_dir = os.path.join(model_path, "eval_test", "ocr_output", "gt", "prediction_jsons")
    render_json_dir = os.path.join(model_path, "eval_test", f"ocr_output", f"renders_{args.iteration}", "prediction_jsons")
    results = evaluate_cer(gt_json_dir, render_json_dir)

    # if args.wandb_flag and wandb.run is None:
    #     wandb.init(project=os.environ["WANDB_PROJECT"],
    #            id=os.environ["WANDB_RUN_ID"],
    #            resume="allow")
    #     wandb.log({
    #         'OCR_CER': results['overall_cer'],
    #         'OCR_WER': results['overall_wer'],
    #         'ocr_eval_iteration': args.iteration,
    #     })

    ocr_results_file = os.path.join(model_path, "eval_test", "ocr_output", "ocr_results.json")
    if os.path.exists(ocr_results_file):
        with open(ocr_results_file, 'r') as f:
            existing_results = json.load(f)
    else:
        existing_results = {}
    existing_results[f"renders_{args.iteration}"] = results
    with open(ocr_results_file, 'w') as f:
        json.dump(existing_results, f, indent=2)

    print("OCR evaluation complete.")