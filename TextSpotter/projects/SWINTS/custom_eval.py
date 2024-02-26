# import json
# import numpy as np
# from tqdm import tqdm
# from pycocotools import mask as coco_mask
# import pickle
# from collections import defaultdict
#
#
# def load_json(filepath):
#     with open(filepath, 'r') as f:
#         return json.load(f)
#
#
# def calculate_iou(bbox1, bbox2, format1="xywh", format2="xywh"):
#     """
#     Calculate the Intersection over Union (IoU) of two bounding boxes.
#
#     Parameters:
#     bbox1, bbox2: list, tuple
#         The coordinates and dimensions of the bounding boxes
#     format1, format2: str
#         The format of the bounding boxes. Either "xywh" or "xyxy".
#
#     Returns:
#     float
#         The IoU of bbox1 and bbox2
#     """
#
#     # Convert bbox1 to xywh format if it is in xyxy format
#     if format1 == "xyxy":
#         x1_1, y1_1, x2_1, y2_1 = bbox1
#         w1 = x2_1 - x1_1
#         h1 = y2_1 - y1_1
#         x1, y1 = x1_1, y1_1
#     else:
#         x1, y1, w1, h1 = bbox1
#
#     # Convert bbox2 to xywh format if it is in xyxy format
#     if format2 == "xyxy":
#         x1_2, y1_2, x2_2, y2_2 = bbox2
#         w2 = x2_2 - x1_2
#         h2 = y2_2 - y1_2
#         x2, y2 = x1_2, y1_2
#     else:
#         x2, y2, w2, h2 = bbox2
#
#     # Calculate the coordinates of the intersection rectangle
#     x1_inter = max(x1, x2)
#     y1_inter = max(y1, y2)
#     x2_inter = min(x1 + w1, x2 + w2)
#     y2_inter = min(y1 + h1, y2 + h2)
#
#     # Calculate the area of the intersection rectangle
#     intersection_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
#
#     # Calculate the area of both bounding boxes
#     bbox1_area = w1 * h1
#     bbox2_area = w2 * h2
#
#     # Calculate the area of the union
#     union_area = bbox1_area + bbox2_area - intersection_area
#
#     # Avoid division by zero
#     if union_area == 0:
#         return 0
#
#     # Calculate IoU
#     iou = intersection_area / union_area
#
#     return iou
#
#
# def compute_accuracy(annotations, results):
#
#     # Initialize overall counters
#     TP = 0
#     FP = 0
#     FN = 0
#
#     # Initialize tag-specific counters
#     tag_metrics = defaultdict(lambda: {'TP': 0, 'FP': 0, 'FN': 0})
#
#
#     overall_correct = 0
#     overall_total = 0
#     tag_stats = {}
#     img_tag_stats = {}
#
#     # process the images attribute of annotations to get the list of tags for each image id
#     image_tags = {}
#     total_images = len(annotations['images'])
#     print(f'total results: {len(results)}')
#     print(f'total images in results: {len(set([res["image_id"] for res in results]))}')
#
#     id_to_fname = {}
#
#     for img in annotations['images']:
#         id_to_fname[img['id']] = img['file_name'].split('.')[0]
#         # if img['id'] already exists, append the tags and make a set - keep only unique
#         if img['id'] in image_tags:
#             image_tags[img['id']] = list(set(image_tags[img['id']] + img['tags']))
#         else:
#             image_tags[img['id']] = img['tags']
#         # and keep track of the total number of images for each tag
#         for tag in img['tags']:
#             if tag in img_tag_stats:
#                 img_tag_stats[tag]['total'] += 1
#             else:
#                 img_tag_stats[tag] = {'correct': 0, 'total': 1}
#
#     total_annos = len(annotations['annotations'])
#     print(f'Total annotations: {total_annos}')
#
#     problems = 0
#     not_problems = 0
#
#     overall_detection = 0
#     detection_tag_stats = {}
#
#     for ann in tqdm(annotations['annotations']):
#
#         # for each annotation, load the appropriate predictions and compare
#         results_folder = "/home/jtyo/YAMTS-perph-te/model_predictions/te_model_predictions/"
#         results_filepath = results_folder + id_to_fname[ann['image_id']] + '.pkl'
#
#         # Open the file in read-binary mode ('rb')
#         try:
#             with open(results_filepath, 'rb') as file:
#                 # Load the data from the file
#                 data = pickle.load(file)
#         except:
#             problems += 1
#             # print(f'Could not find results:')
#             # print(f'\tFilepath: {results_filepath}')
#             # print(f'\timage_id: {ann["image_id"]}')
#             # print(f'\timage fn: {id_to_fname[ann["image_id"]]}')
#             continue
#
#         not_problems += 1
#
#         overall_total += 1
#         image_id = ann['image_id']
#         gt_text = ann['attributes']['transcription']  # replace with your field name for ground truth text
#         tags = image_tags[image_id]
#
#         for tag in tags:
#             if tag not in detection_tag_stats:
#                 detection_tag_stats[tag] = {'correct': 0, 'total': 1}
#             else:
#                 detection_tag_stats[tag]['total'] += 1
#
#         anno_bbox = ann['bbox']
#         found_match = False
#
#         for bx in data.get('bbox', []):
#             iou = calculate_iou(anno_bbox, bx[:4], format1='xywh', format2='xyxy')
#             if iou > 0.5:
#                 TP += 1  # Overall True Positive
#                 found_match = True
#
#                 # Tag-specific True Positives
#                 for tag in tags:
#                     tag_metrics[tag]['TP'] += 1
#
#                 break  # Exit the loop if we find a match
#
#         # Overall False Negatives
#         if not found_match:
#             FN += 1
#
#             # Tag-specific False Negatives
#             for tag in tags:
#                 tag_metrics[tag]['FN'] += 1
#
#         # Count overall and tag-specific False Positives
#         for data_bbox in data.get('bbox', []):
#             found_match = False
#             for anno_bbox in annotations['annotations']:
#                 iou = calculate_iou(anno_bbox['bbox'], data_bbox[:4], format1='xywh', format2='xyxy')
#                 if iou > 0.5:
#                     found_match = True
#                     break  # Exit the loop if we find a match
#
#         # Overall False Positives
#         if not found_match:
#             FP += 1
#
#         # Tag-specific False Positives
#         for tag in tags:
#             tag_metrics[tag]['FP'] += 1
#
#         # now get the predictions for this image
#         preds = data.get('number_pred', None)
#         for tag in tags:
#             if tag in tag_stats:
#                 tag_stats[tag]['total'] += 1
#             else:
#                 tag_stats[tag] = {'correct': 0, 'total': 1}
#
#         if preds is None:
#             continue
#         if gt_text.strip() in preds:
#             # this is a correct prediction
#             overall_correct += 1
#             # now add the tag statistics
#             for tag in tags:
#                 if tag not in tag_stats:
#                     tag_stats[tag] = {'correct': 0, 'total': 1}
#                 tag_stats[tag]['correct'] += 1
#
#     # Compute overall accuracy
#     overall_accuracy = overall_correct / overall_total if overall_total > 0 else 0
#
#     # Compute accuracy for each tag
#     tag_accuracies = {}
#     for tag, stats in tag_stats.items():
#         tag_accuracies[tag] = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
#
#     print('problems: ', problems)
#     print('not problems: ', not_problems)
#     print('% problems ', problems / (problems + not_problems) * 100)
#
#     # Compute overall metrics
#     precision = TP / (TP + FP) if TP + FP > 0 else 0
#     recall = TP / (TP + FN) if TP + FN > 0 else 0
#     f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
#
#     print(f'Overall Precision: {precision:.2f}')
#     print(f'Overall Recall: {recall:.2f}')
#     print(f'Overall F1 Score: {f1_score:.2f}')
#
#     # Compute tag-specific metrics
#     for tag, metrics in tag_metrics.items():
#         tag_precision = metrics['TP'] / (metrics['TP'] + metrics['FP']) if metrics['TP'] + metrics['FP'] > 0 else 0
#         tag_recall = metrics['TP'] / (metrics['TP'] + metrics['FN']) if metrics['TP'] + metrics['FN'] > 0 else 0
#         tag_f1_score = 2 * (tag_precision * tag_recall) / (
#                     tag_precision + tag_recall) if tag_precision + tag_recall > 0 else 0
#         print(f'Precision for tag {tag}: {tag_precision:.2f}')
#         print(f'Recall for tag {tag}: {tag_recall:.2f}')
#         print(f'F1 Score for tag {tag}: {tag_f1_score:.2f}')
#
#     return overall_accuracy, tag_accuracies, tag_stats, total_annos, total_images, img_tag_stats, detection_tag_stats, overall_detection
#
# if __name__ == '__main__':
#     annotation_filepath = '/home/jtyo/Repos/PersonalRepos/SwinTextSpotter/data/perph/perph_te_tagged.json'
#     # result_filepath = '/home/jtyo/swints_models/swints_base_output_tt_1/inference/text_results.json'
#     result_filepath = '/home/jtyo/swints_models/inference/text_results.json'
#     # other potential results file to use: temp_all_det_cors.txt
#
#     annotations = load_json(annotation_filepath)
#     results = load_json(result_filepath)
#
#     overall_accuracy, tag_accuracies, tag_info, total_annos, total_imgs, img_tag_stats, detection_tag_stats, overall_detection = compute_accuracy(annotations, results)
#
#     print(f'Overall Accuracy: {overall_accuracy * 100:.2f}%')
#     print(f'Overall Detection Accuracy: {overall_detection / total_annos * 100:.2f}%')
#     for tag, acc in tag_accuracies.items():
#         print(f'Accuracy for tag {tag}: {acc * 100:.2f}%, Total Correct: {tag_info[tag]["correct"]}, Total: {tag_info[tag]["total"]}, % imgs with tag: {img_tag_stats[tag]["total"] / total_imgs * 100:.2f}%, % annos with tag: {tag_info[tag]["total"] / total_annos * 100:.2f}%, % detections with tag: {detection_tag_stats[tag]["correct"] / detection_tag_stats[tag]["total"] * 100:.2f}%')


import json
import pickle
from collections import defaultdict

def load_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def calculate_iou(bbox1, bbox2, format1="xywh", format2="xywh"):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters:
    bbox1, bbox2: list, tuple
        The coordinates and dimensions of the bounding boxes
    format1, format2: str
        The format of the bounding boxes. Either "xywh" or "xyxy".

    Returns:
    float
        The IoU of bbox1 and bbox2
    """

    # Convert bbox1 to xywh format if it is in xyxy format
    if format1 == "xyxy":
        x1_1, y1_1, x2_1, y2_1 = bbox1
        w1 = x2_1 - x1_1
        h1 = y2_1 - y1_1
        x1, y1 = x1_1, y1_1
    else:
        x1, y1, w1, h1 = bbox1

    # Convert bbox2 to xywh format if it is in xyxy format
    if format2 == "xyxy":
        x1_2, y1_2, x2_2, y2_2 = bbox2
        w2 = x2_2 - x1_2
        h2 = y2_2 - y1_2
        x2, y2 = x1_2, y1_2
    else:
        x2, y2, w2, h2 = bbox2

    # Calculate the coordinates of the intersection rectangle
    x1_inter = max(x1, x2)
    y1_inter = max(y1, y2)
    x2_inter = min(x1 + w1, x2 + w2)
    y2_inter = min(y1 + h1, y2 + h2)

    # Calculate the area of the intersection rectangle
    intersection_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)

    # Calculate the area of both bounding boxes
    bbox1_area = w1 * h1
    bbox2_area = w2 * h2

    # Calculate the area of the union
    union_area = bbox1_area + bbox2_area - intersection_area

    # Avoid division by zero
    if union_area == 0:
        return 0

    # Calculate IoU
    iou = intersection_area / union_area

    return iou

def compute_metrics(TP, FP, FN):
    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    return precision, recall, f1_score


def compute_accuracy(annotations):
    bbox_stats = {'TP': 0, 'FP': 0, 'FN': 0}
    e2e_stats = {'TP': 0, 'FP': 0, 'FN': 0, 'correct_text': 0, 'total_text': 0}
    bbox_tag_stats = defaultdict(lambda: {'TP': 0, 'FP': 0, 'FN': 0, 'correct_text': 0, 'total_text': 0})
    e2e_tag_stats = defaultdict(lambda: {'TP': 0, 'FP': 0, 'FN': 0, 'correct_text': 0, 'total_text': 0})

    # Initialize 'none' category
    bbox_tag_stats['none'] = {'TP': 0, 'FP': 0, 'FN': 0, 'correct_text': 0, 'total_text': 0}
    e2e_tag_stats['none'] = {'TP': 0, 'FP': 0, 'FN': 0, 'correct_text': 0, 'total_text': 0}

    image_tags = {img['id']: img['tags'] for img in annotations['images']}
    total_imgs = len(image_tags.keys())
    total_tags = {tag: 0 for img in annotations['images'] for tag in img['tags']}

    none_count = 0  # Initialize counter for "none" tags
    for img_id, tg in image_tags.items():
        if not tg:  # Check if tag list is empty
            none_count += 1
            tg.append('none')  # Add 'none' to tags

        for tag in tg:
            total_tags[tag] = total_tags.get(tag, 0) + 1

    # Include the 'none' count in the statistics
    total_tags['none'] = none_count

    # now print out count and % of images with each tag
    for tag, count in total_tags.items():
        print(f'{tag}: {count} ({count / total_imgs * 100:.2f}%)')
    print('========================================')

    id_to_fname = {img['id']: img['file_name'].split('.')[0] for img in annotations['images']}

    file_not_found = 0

    # Group annotations by image_id
    annotations_by_image = defaultdict(list)
    for ann in annotations['annotations']:
        annotations_by_image[ann['image_id']].append(ann)

    for image_id, image_annotations in annotations_by_image.items():
        # Load prediction data
        results_filepath = f"/home/jtyo/YAMTS-perph-te/off-the-shelf/pretrained_model_predictions/te_model_predictions/{id_to_fname[image_id]}.pkl"

        try:
            with open(results_filepath, 'rb') as file:
                data = pickle.load(file)
        except FileNotFoundError:
            file_not_found += 1
            continue

        # Initialize predictions_matched list for this image
        predictions_matched = [False for _ in data.get('bbox', [])]

        for ann in image_annotations:
            e2e_stats['total_text'] += 1

            gt_text = ann['attributes']['transcription']
            tags = image_tags[ann['image_id']]
            if not tags:  # Check if tag list is empty
                tags.append('none')  # Add 'none' to tags
            anno_bbox = ann['bbox']

            detection_matched = False
            e2e_matched = False

            for tag in tags:
                bbox_tag_stats[tag]['total_text'] += 1
                e2e_tag_stats[tag]['total_text'] += 1

            for pred_index, pred_bbox in enumerate(data.get('bbox', [])):
                iou = calculate_iou(ann['bbox'], pred_bbox[:4], format1='xywh', format2='xyxy')

                if iou > 0.25:
                    detection_matched = True
                    bbox_stats['TP'] += 1
                    predictions_matched[pred_index] = True

                    for tag in tags:
                        bbox_tag_stats[tag]['TP'] += 1

                    preds = data.get('number_pred', None)
                    if preds and gt_text.strip() in preds:
                        e2e_stats['TP'] += 1
                        e2e_stats['correct_text'] += 1
                        e2e_matched = True
                        for tag in tags:
                            e2e_tag_stats[tag]['TP'] += 1
                    break

            if not detection_matched:
                bbox_stats['FN'] += 1
                for tag in tags:
                    bbox_tag_stats[tag]['FN'] += 1

            if not e2e_matched:
                e2e_stats['FN'] += 1
                for tag in tags:
                    e2e_tag_stats[tag]['FN'] += 1

        for matched in predictions_matched:
            if not matched:
                bbox_stats['FP'] += 1
                e2e_stats['FP'] += 1
                for tag in image_tags[image_id]:
                    bbox_tag_stats[tag]['FP'] += 1
                    e2e_tag_stats[tag]['FP'] += 1

    # Compute overall metrics
    bbox_precision, bbox_recall, bbox_f1 = compute_metrics(bbox_stats['TP'], bbox_stats['FP'], bbox_stats['FN'])
    e2e_precision, e2e_recall, e2e_f1 = compute_metrics(e2e_stats['TP'], e2e_stats['FP'], e2e_stats['FN'])
    overall_text_accuracy = e2e_stats['correct_text'] / e2e_stats['total_text'] if e2e_stats['total_text'] > 0 else 0

    print(f'File not found: {file_not_found}')
    print(f'Overall BBox Precision: {bbox_precision:.3f}')
    print(f'Overall BBox Recall: {bbox_recall:.3f}')
    print(f'Overall BBox F1 Score: {bbox_f1:.3f}')
    print(f'Overall Text Accuracy: {overall_text_accuracy:.3f}')
    print(f'Overall E2E Precision: {e2e_precision:.3f}')
    print(f'Overall E2E Recall: {e2e_recall:.3f}')
    print(f'Overall E2E F1 Score: {e2e_f1:.3f}')
    print('========================================')

    # Compute tag-based metrics
    for (tag, stats), (e2e_tag, e2e_stats) in zip(bbox_tag_stats.items(), e2e_tag_stats.items()):
        tag_precision, tag_recall, tag_f1 = compute_metrics(stats['TP'], stats['FP'], stats['FN'])

        e2e_tag_precision, e2e_tag_recall, e2e_tag_f1 = compute_metrics(e2e_stats['TP'], e2e_stats['FP'], e2e_stats['FN'])
        e2e_tag_text_accuracy = e2e_stats['correct_text'] / e2e_stats['total_text'] if e2e_stats['total_text'] > 0 else 0

        print(f"{tag} BBox Precision: {tag_precision:.3f}")
        print(f"{tag} BBox Recall: {tag_recall:.3f}")
        print(f"{tag} BBox F1 Score: {tag_f1:.3f}")
        print(f"{tag} Text Accuracy: {e2e_tag_text_accuracy:.3f}")
        print(f"{tag} E2E Precision: {e2e_tag_precision:.3f}")
        print(f"{tag} E2E Recall: {e2e_tag_recall:.3f}")
        print(f"{tag} E2E F1 Score: {e2e_tag_f1:.3f}")
        print('========================================')


if __name__ == '__main__':
    annotations = load_json('/home/jtyo/Repos/PersonalRepos/SwinTextSpotter/data/perph/perph_te_tagged.json')
    compute_accuracy(annotations)
