import json
import sys
import os


# Function to sort points in clockwise order
def sort_points_clockwise_image_coord(points):
    import math

    # Calculate centroid
    cx = sum(x for x, y in points) / len(points)
    cy = sum(y for x, y in points) / len(points)

    # Sort points based on the angle with the centroid
    # Note that the y-coordinates are subtracted from cy (instead of cy from y)
    # to account for the image coordinate system.
    sorted_points = sorted(points, key=lambda point: -math.atan2(point[1] - cy, point[0] - cx))

    return sorted_points


coco_path = sys.argv[1]
icdar_path = sys.argv[2]

# Load COCO annotations
with open(coco_path, "r") as f:
    coco_data = json.load(f)

print(f'num images: {len(coco_data["images"])}')

# Create a dictionary to map COCO image IDs to file names
coco_id_to_filename = {}
coco_filename_to_id = {}
for image in coco_data["images"]:
    image_id = image["id"]
    filename = image["file_name"]
    coco_id_to_filename[image_id] = filename
    coco_filename_to_id[filename] = image_id

# Create a dictionary to map image_id to a list of segmentation_maps linking to text annotations
coco_id_to_segmentation_maps = {}
for annotation in coco_data["annotations"]:
    image_id = annotation["image_id"]
    segmentation_map = annotation["segmentation"]
    if image_id not in coco_id_to_segmentation_maps:
        coco_id_to_segmentation_maps[image_id] = []
    coco_id_to_segmentation_maps[image_id].append({annotation["attributes"]["transcription"]: segmentation_map})

# Write the annotations
for image_id, segmentation_maps in coco_id_to_segmentation_maps.items():
    print(image_id)
    print(coco_id_to_segmentation_maps[image_id])
    print('=================')
    # filename = coco_id_to_filename[image_id]
    # filename = filename.split(".")[0]
    filename = str(image_id).zfill(7) + ".txt"

    with open(os.path.join(icdar_path, filename), "w") as f:
        for segmentation_map in segmentation_maps:
            for transcription, segmentation in segmentation_map.items():

                # Extract points and sort them
                raw_points = segmentation[0]
                points = [(raw_points[i], raw_points[i + 1]) for i in range(0, len(raw_points), 2)]
                # print(points)
                sorted_points = sort_points_clockwise_image_coord(points)
                # print(sorted_points)
                # print('=================')

                # Write sorted points to file
                for x, y in sorted_points:
                    f.write(f"{round(x)},{round(y)},")
                f.write("####")
                f.write(transcription)
                f.write("\n")
