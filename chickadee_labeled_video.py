import re

import cropped_predictions as cp

from tqdm import tqdm
from pathlib import Path

lp_dir = Path("/home/ksikka/synced/lightning-pose")

outputs_dir = lp_dir / "outputs/chickadee/cropzoom"

data_dir = outputs_dir / "detector_0" / "cropped_videos"
video_dir_1 = outputs_dir / "pose_supervised_0" / "video_preds" / "labeled_videos"
video_dir_2 = outputs_dir / "pose_ctx_0" / "video_preds" / "labeled_videos"

for video_file in list(video_dir_2.iterdir()):
    print(str(video_file))
    video_file_og = data_dir / re.sub(r"_labeled\.", ".", video_file.name)
    print(str(video_file_og))
    preds_df_path = video_file.parent.parent / (video_file_og.stem + ".csv")
    print(str(preds_df_path))
    preds_df = cp.read_preds_file(preds_df_path)
    cp.process_video(str(video_file_og), preds_df, "/tmp/ctx/" + video_file.name)