import json
import sys
import os

coco_path = sys.argv[1]

# Load COCO annotations
with open(coco_path, "r") as f:
    coco_data = json.load(f)

print(f'num images: {len(coco_data["images"])}')
print(f'num annotations: {len(coco_data["annotations"])}')

numbers_with_letters = 0
for annotation in coco_data["annotations"]:
    transcription = annotation["attributes"]["transcription"]
    if any(char.isdigit() for char in transcription) and any(char.isalpha() for char in transcription):
        numbers_with_letters += 1

print(f'num numbers with letters: {numbers_with_letters}')
print(f'% numbers with letters: {numbers_with_letters / len(coco_data["annotations"])}')



# (base) jtyo@Sympathy:~/Repos/PersonalRepos/SwinTextSpotter$ python projects/SWINTS/dataset_utils/dataset_stats.py data/perph/perph_tr.json
# num images: 1928
# num annotations: 4443

# python projects/SWINTS/dataset_utils/dataset_stats.py data/perph/perph_te.json
# num images: 483
# num annotations: 1135

# Total
#
#