import json
import sys

coco_file_path = sys.argv[1]
output_path = sys.argv[2]


# Your existing COCO file
# coco_file_path = 'path/to/your_coco_annotations.json'

# The character mapping
cV2 = [' ','!','"','#','$','%','&','\'','(',')','*','+',',','-','.','/','0','1','2','3','4','5','6','7','8','9',':',';','<','=','>','?','@','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','[','\\',']','^','_','`','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','{','|','}','~','´', "~", "ˋ", "ˊ","﹒", "ˀ", "˜", "ˇ", "ˆ", "˒","‑"]

# Read your existing COCO annotations
with open(coco_file_path, 'r') as f:
    coco_data = json.load(f)

# Function to convert the text into numerical representations
def convert_text(text):
    max_len = 10
    recs = [0 for _ in range(max_len)]
    for ix, ict in enumerate(text):
        if ix >= max_len:
            continue
        if ict in cV2:
            recs[ix] = cV2.index(ict) + 1
        else:
            recs[ix] = 0
    return recs

# Modify the annotations as needed
for annotation in coco_data['annotations']:
    # bbox = annotation['bbox']
    # width = max(0, bbox[2] - bbox[0] + 1)
    # height = max(0, bbox[3] - bbox[1] + 1)
    # xmin, ymin = bbox[:2]
    # xmax, ymax = xmin + width, ymin + height

    # Here, you can add code to compute segmentation if needed

    # annotation['bbox'] = [xmin, ymin, width, height]
    # annotation['area'] = width * height

    # If you have the text stored in some other field in your original COCO, adjust as needed
    text = annotation['attributes']['transcription']
    annotation['rec'] = convert_text(text)
    
coco_data['categories'] = [{
      'id': 1,
      'name': 'text',
      'supercategory': 'beverage',
      'keypoints': ['mean',
                    'xmin',
                    'x2',
                    'x3',
                    'xmax',
                    'ymin',
                    'y2',
                    'y3',
                    'ymax',
                    'cross']  # only for BDN
  }]

# You can add any additional modifications required for categories, images, etc.

# Save the modified COCO data
# output_path = 'path/to/modified_annotations.json'
with open(output_path, 'w') as f:
    json.dump(coco_data, f)

