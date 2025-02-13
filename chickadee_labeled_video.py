import re

import cropped_predictions as cp

from tqdm import tqdm
from pathlib import Path

lp_dir = Path("/home/ksikka/synced/lightning-pose")

outputs_dir = lp_dir / "outputs/chickadee/cropzoom"

data_dir = outputs_dir / "detector_0" / "cropped_videos"
video_dir_1 = outputs_dir / "pose_supervised_2" / "video_preds" / "labeled_videos"
video_dir_2 = outputs_dir / "pose_ctx_2" / "video_preds" / "labeled_videos"

output_vid_dir1 = "/tmp/supervised2/"
output_vid_dir2 = "/tmp/ctx2/"

def process_video(video_file, output_vid_dir):
    video_file_og = data_dir / re.sub(r"_labeled\.", ".", video_file.name)
    preds_df_path = video_file.parent.parent / (video_file_og.stem + ".csv")
    preds_df = cp.read_preds_file(preds_df_path)
    cp.process_video(str(video_file_og), preds_df, output_vid_dir + video_file.name)

def iterate_matching_files(dir1_path, dir2_path):
    """
    Iterates over matching files in two directories.

    Args:
      dir1_path: Path to the first directory.
      dir2_path: Path to the second directory.

    Yields:
      Tuples of (path1, path2) for each pair of matching files.
    """

    dir1 = Path(dir1_path)
    dir2 = Path(dir2_path)

    for file1 in dir1.iterdir():
        if file1.is_file():
            file2 = dir2 / file1.name
            if file2.is_file():
                yield (file1, file2)

list(map(lambda x: process_video(x, output_vid_dir1), video_dir_1.iterdir()))
list(map(lambda x: process_video(x, output_vid_dir2), video_dir_2.iterdir()))

for file1, file2 in iterate_matching_files(output_vid_dir1, output_vid_dir2):
    print(file1, file2)
    combined_path = "/tmp/combined2/" + file1.name
    cp.combine_videos_side_by_side(str(file1), str(file2), combined_path)
