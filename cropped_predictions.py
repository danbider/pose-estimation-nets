#!/usr/bin/env python
# coding: utf-8

# In[94]:


import pandas as pd
import os
from PIL import Image, ImageDraw, ImageColor
from pathlib import Path

"""Image processing part"""

# TODO read from a file?
skeleton = [
    ("botBeak", "topBeak"),
    ("topBeak", "topHead"),
    ("topHead", "rightEye"),
    ("topHead", "leftEye"),
    ("topHead", "backHead"),
    ("backHead", "centerBack"),
    ("centerBack", "leftWing"),
    ("centerBack", "rightWing"),
    ("centerBack", "baseTail"),
    ("baseTail", "tipTail"),
    ("centerBack", "leftAnkle"),
    ("centerBack", "rightAnkle"),
    ("centerBack", "centerChes"),
    ("centerBack", "leftNeck"),
    ("centerBack", "rightNeck"),
    ("leftAnkle", "leftFoot"),
    ("rightAnkle", "rightFoot"),
]

def annotate_image_with_predictions(
    img, predictions, likelihood_threshold, skeleton_color
):
    """
    Annotates an image with dots at specific points.
    Args:
        root_dir (str): Path to the directory containing image files.
        image_path (str): Path to the image file relative to root_dir.
        predictions (pd.Dataframe): A pandas dataframe where index = keypoint
                                    and columns are x, y, likelihood
                                    representing the points.
    """
    draw = ImageDraw.Draw(img, "RGBA")
    dot_color = ImageColor.getrgb(skeleton_color) + (255,)
    skeleton_color = ImageColor.getrgb(skeleton_color) + (128,)

    # Draw dots on the image
    for i, (keypoint, (x, y, likelihood)) in enumerate(predictions.iterrows()):
        # Skip points below the threshold
        if likelihood < likelihood_threshold:
            continue
        draw.ellipse(
            (x - 2, y - 2, x + 2, y + 2), fill=dot_color
        )  # Draw a dot with radius 2

    for keypoint1, keypoint2 in skeleton:
        pred1 = predictions.loc[keypoint1]
        pred2 = predictions.loc[keypoint2]
        x1, y1, likelihood1 = pred1.x, pred1.y, pred1.likelihood
        x2, y2, likelihood2 = pred2.x, pred2.y, pred2.likelihood
        if min(likelihood1, likelihood2) < likelihood_threshold:
            continue
        draw.line([(x1, y1), (x2, y2)], fill=skeleton_color, width=2)



def read_preds_file(preds_file):
    preds_df = pd.read_csv(preds_file, header=[0, 1, 2], index_col=0)
    preds_df = preds_df.droplevel(level=0, axis=1)  # axis=1 for columns
    preds_df = preds_df.loc[
        :, preds_df.columns.get_level_values(1).isin(["x", "y", "likelihood"])
    ]
    return preds_df



def draw_bbox(img, bbox):
    draw = ImageDraw.Draw(img)
    x0, y0, x1, y1 = bbox.x, bbox.y, bbox.x + bbox.w, bbox.y + bbox.h

    color = "black"
    dot_spacing = 5
    dot_length = 2

    # Clip bbox to image boundaries
    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(img.width, x1)
    y1 = min(img.height, y1)

    # Draw top and bottom lines
    for x in range(x0, x1, dot_spacing + dot_length):
        draw.line(
            [(x, y0), (min(x + dot_length, x1), y0)], fill=color
        )  # min to prevent going over the limit
        draw.line([(x, y1), (min(x + dot_length, x1), y1)], fill=color)

    # Draw left and right lines
    for y in range(y0, y1, dot_spacing + dot_length):
        draw.line([(x0, y), (x0, min(y + dot_length, y1))], fill=color)
        draw.line([(x1, y), (x1, min(y + dot_length, y1))], fill=color)

    padding = int(bbox.w * 0.125)
    x0_pad = max(0, x0 - padding)
    y0_pad = max(0, y0 - padding)
    x1_pad = min(img.width, x1 + padding)
    y1_pad = min(img.height, y1 + padding)

    return img.crop((x0_pad, y0_pad, x1_pad, y1_pad))


class Dataset:

    def read_files(dataset):
        bbox_file, labels_file, single_preds_file, pose_preds_file = (
            dataset.bbox_file,
            dataset.labels_file,
            dataset.single_preds_file,
            dataset.pose_preds_file,
        )
        dataset.bbox_df = pd.read_csv(bbox_file, header=[0], index_col=0)
        dataset.labels_df = read_preds_file(labels_file)
        dataset.single_preds_df = read_preds_file(single_preds_file)
        dataset.pose_preds_df = read_preds_file(pose_preds_file)
        assert set(dataset.single_preds_df.index) == set(dataset.pose_preds_df.index)
        assert set(dataset.bbox_df.index) == set(dataset.pose_preds_df.index)


    def generate_annotated_image(dataset, img_path):
        labels_df, single_preds_df, pose_preds_df, bbox_df = dataset.labels_df, dataset.single_preds_df, dataset.pose_preds_df, dataset.bbox_df
        labels = labels_df.loc[img_path].unstack(level=1)
        labels["likelihood"] = 1.0
        red_preds = single_preds_df.loc[img_path].unstack(level=1)
        green_preds = pose_preds_df.loc[img_path].unstack(level=1)
        bbox = bbox_df.loc[img_path]

        abs_img_path = Path(dataset.data_dir) / img_path
        red_img = Image.open(abs_img_path)
        annotate_image_with_predictions(red_img, labels, 0.9, "#1E90FF")
        annotate_image_with_predictions(red_img, red_preds, 0.9, "#FF4500")
        red_img = draw_bbox(red_img, bbox)


        green_img = Image.open(abs_img_path)
        annotate_image_with_predictions(green_img, labels, 0.9, "#1E90FF")
        annotate_image_with_predictions(green_img, green_preds, 0.9, "#ADFF2F")
        green_img = draw_bbox(green_img, bbox)

        return (red_img, green_img)



def create_image_gallery_html(image_dir, parent_dir=None):
  """
  Creates an HTML file that displays images from a directory and links to subfolders.

  Args:
    image_dir: Path to the directory containing images.
    parent_dir: Path to the parent directory (used for creating links to subfolders).
  """

  html = """
  <!DOCTYPE html>
  <html>
  <head>
  <title>Image Gallery</title>
  <style>
  .img-container {
      display: inline-block;
      margin: 10px;
      width: 50%; /* Make images take up 50% of the container width */
      box-sizing: border-box; /* Include padding and border in the element's total width and height */
    }
    img {
      width: 100%;
      height: auto;
    }
  .pair-container {
      display: flex;
      align-items: center;
      width: 100%; /* Make pair containers take up full width */
      max-width: 1200px;
    }
  </style>
  </head>
  <body>
  <h1>Image Gallery</h1>
  """

  if parent_dir:
    html += f'<p><a href="../index.html">Parent Directory</a></p>'

  # List subfolders
  subfolders = [f for f in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, f))]
  if subfolders:
    html += '<h2>Subfolders</h2><ul>'
    for subfolder in subfolders:
      html += f'<li><a href="{subfolder}/index.html">{subfolder}</a></li>'
    html += '</ul>'

  html += '<div class="gallery">'

  # List images
  # Identify and pair red and green images
  image_pairs = {}
  for filename in os.listdir(image_dir):
    if filename.endswith(('_red.png', '_green.png')):
      # Extract root name (without _red or _green)
      root_name = filename.replace('_red.png', '').replace('_green.png', '')
      if root_name not in image_pairs:
        image_pairs[root_name] = {}
      if filename.endswith('_red.png'):
        image_pairs[root_name]['red'] = filename
      else:
        image_pairs[root_name]['green'] = filename

  # Display paired images
  for root_name, pair in image_pairs.items():
    if 'red' in pair and 'green' in pair:
      html += f"""
      <div class="pair-container">
        <div class="img-container"><img src="{pair['red']}" alt="{root_name}_red"></div>
        <div class="img-container"><img src="{pair['green']}" alt="{root_name}_green"></div>
      </div>
      """

  html += """
  </div>
  </body>
  </html>
  """

  with open(os.path.join(image_dir, 'index.html'), 'w') as f:
    f.write(html)

  # Recursively create HTML for subfolders
  for subfolder in subfolders:
    create_image_gallery_html(os.path.join(image_dir, subfolder), image_dir)



"""Video processing part"""

import cv2
import pandas as pd
KEYPOINT_COLORS = {
    "botBeak": (255, 255, 100),    # Bright Yellow
    "topBeak": (255, 255, 0),    # Brighter Yellow

    "topHead": (100, 100, 255),      # Bright Blue
    "rightEye": (200, 200, 255),  # Brighter Light Blue
    "leftEye": (100, 100, 255),   # Brighter Darker Blue
    "backHead": (50, 50, 255),     # Brighter Dark Blue

    "centerBack": (0, 255, 0),    # Bright Green
    "leftWing": (128, 255, 128),  # Light Green (already bright)
    "rightWing": (0, 200, 0),     # Brighter Green
    "baseTail": (0, 255, 0),     # Bright Green (same as center back)
    "tipTail": (0, 150, 0),       # Brighter Dark Green

    "leftAnkle": (255, 100, 100),   # Bright Brown
    "leftFoot": (200, 0, 0),     # Brighter Dark Brown
    "rightAnkle": (200, 0, 0),    # Brighter Maroon
    "rightFoot": (150, 0, 0),      # Brighter Very Dark Brown

    "centerChes": (255, 200, 0),  # Bright Orange
    "leftNeck": (255, 165, 0),    # Brighter Light Orange
    "rightNeck": (255, 140, 0),    # Brighter Darker Orange
}
import numpy as np
# Convert the colors to BGR
for keypoint, color in KEYPOINT_COLORS.items():
    KEYPOINT_COLORS[keypoint] = tuple(reversed(color))

def draw_keypoints_on_frame(frame, df, threshold=0.5):
    """
    Overlays keypoints on a single frame of a video.

    Args:
      frame: The image frame (NumPy array) to draw on.
      df: Pandas DataFrame with keypoint data for the frame.
            Each row should be a keypoint with columns 'x', 'y', and 'likelihood'.
      threshold: The likelihood threshold for drawing a keypoint.
    """
    for keypoint, row in df.iterrows():
        x, y, likelihood = row['x'], row['y'], row['likelihood']
        if likelihood > threshold:
            color = KEYPOINT_COLORS.get(keypoint)
            if color:
                cv2.circle(frame, (int(x), int(y)), 3, color, -1)
                # Draw skeleton
    for keypoint1, keypoint2 in skeleton:
        pred1 = df.loc[keypoint1]
        pred2 = df.loc[keypoint2]
        x1, y1, likelihood1 = pred1['x'], pred1['y'], pred1['likelihood']
        x2, y2, likelihood2 = pred2['x'], pred2['y'], pred2['likelihood']
        if min(likelihood1, likelihood2) > threshold:
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (47,255,173),
                     2)  # White color for skeleton

import cv2
def process_video(video_path, df, output_path, threshold=0.9):
    """
    Overlays keypoints on a video.

    Args:
      video_path: Path to the video file.
      df: Pandas DataFrame with keypoint data for the entire video.
            Each row should be a frame, with columns as tuples: (keypoint, x | y | likelihood).
      threshold: The likelihood threshold for drawing a keypoint.
    """

    assert Path(video_path).is_file()
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter.fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    for frame_idx in range(len(df)):
        ret, frame = cap.read()
        if not ret:
            break

        # Prepare data for the current frame
        frame_df = df.iloc[frame_idx].unstack(level=1)

        # Draw keypoints on the frame
        draw_keypoints_on_frame(frame, frame_df, threshold)

        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

def combine_videos_side_by_side(video_path1, video_path2, output_path):
    """
    Combines two videos side-by-side into a single video.

    Args:
      video_path1: Path to the first video file.
      video_path2: Path to the second video file.
      output_path: Path to save the combined video file.
    """

    cap1 = cv2.VideoCapture(video_path1)
    cap2 = cv2.VideoCapture(video_path2)

    frame_width1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps1 = int(cap1.get(cv2.CAP_PROP_FPS))

    frame_width2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps2 = int(cap2.get(cv2.CAP_PROP_FPS))

    # Check if both videos have the same FPS and height
    if fps1!= fps2:
        print("Error: Videos have different FPS. Cannot combine.")
        return
    if frame_height1!= frame_height2:
        print("Error: Videos have different heights. Cannot combine.")
        return

    # Calculate output video dimensions
    output_width = frame_width1 + frame_width2
    output_height = max(frame_height1, frame_height2)  # Use the larger height

    fourcc = cv2.VideoWriter.fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps1, (output_width, output_height))

    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if not ret1 or not ret2:
            break

        # Combine frames side-by-side
        combined_frame = cv2.hconcat([frame1, frame2])

        out.write(combined_frame)

    cap1.release()
    cap2.release()
    out.release()
    cv2.destroyAllWindows()