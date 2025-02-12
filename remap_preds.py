from lightning_pose.utils.cropzoom import generate_cropped_csv_file
from tqdm import tqdm

output_dir = "/home/ks3582/synced/outputs/chickadee/cropzoom/"
preds_csv = output_dir + "pose_ctx_{idx}/image_preds/cropped_CollectedData_merged{new_suffix}.csv/predictions.csv"
bbox_csv = output_dir + "detector_{idx}/image_preds/CollectedData_merged{new_suffix}.csv/bbox.csv"
out_preds_csv = output_dir + "pose_ctx_{idx}/image_preds/cropped_CollectedData_merged{new_suffix}.csv/remapped_predictions.csv"

for idx in tqdm(range(3)):
    for new_suffix in tqdm(["", "_new"]):
        generate_cropped_csv_file(
            input_csv_file=preds_csv.format(new_suffix=new_suffix, idx=idx),
            input_bbox_file=bbox_csv.format(new_suffix=new_suffix, idx=idx),
            output_csv_file=out_preds_csv.format(new_suffix=new_suffix, idx=idx),
            mode="add",
        )
