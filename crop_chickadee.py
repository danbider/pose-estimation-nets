import cropped_predictions as cp

from tqdm import tqdm
from pathlib import Path

lp_dir = Path("/home/ks3582/synced/lightning-pose")
data_dir = lp_dir / "data" / "chickadee"
outputs_dir = Path("/home/ks3582/synced/outputs/chickadee/cropzoom")

ind = cp.Dataset()
ind.data_dir = Path(data_dir)
ind.labels_file = lp_dir / "CollectedData_merged.csv"
ind.single_preds_file = (
    outputs_dir
    / "pose_supervised_2/image_preds/cropped_CollectedData_merged.csv/remapped_predictions.csv"
)
ind.pose_preds_file = (
    outputs_dir
    / "pose_ctx_2/image_preds/cropped_CollectedData_merged.csv/remapped_predictions.csv"
)
ind.bbox_file = outputs_dir / "detector_0/image_preds/CollectedData_merged.csv/bbox.csv"
ind.read_files()

ood = cp.Dataset()
ood.data_dir = Path(data_dir)
ood.labels_file = lp_dir / "CollectedData_merged_new.csv"
ood.single_preds_file = (
    outputs_dir
    / "pose_supervised_2/image_preds/cropped_CollectedData_merged_new.csv/remapped_predictions.csv"
)
ood.pose_preds_file = (
    outputs_dir
    / "pose_ctx_2/image_preds/cropped_CollectedData_merged_new.csv/remapped_predictions.csv"
)
ood.bbox_file = (
    outputs_dir / "detector_0/image_preds/CollectedData_merged_new.csv/bbox.csv"
)
ood.read_files()

output_dir = Path("/home/ks3582/pose_vs_ctx_gallery/")

# Previews
for img_path in tqdm(ind.single_preds_df.index):
    red, green = ind.generate_annotated_image(img_path)

    img_path = Path(img_path)

    red_path = output_dir / "ind" / img_path.with_stem(img_path.stem + "_red")
    red_path.parent.mkdir(parents=True, exist_ok=True)
    red.save(red_path)

    green_path = output_dir / "ind" / img_path.with_stem(img_path.stem + "_green")
    green_path.parent.mkdir(parents=True, exist_ok=True)
    green.save(green_path)


for img_path in tqdm(ood.single_preds_df.index):
    red, green = ood.generate_annotated_image(img_path)

    img_path = Path(img_path)

    red_path = output_dir / "ood" / img_path.with_stem(img_path.stem + "_red")
    red_path.parent.mkdir(parents=True, exist_ok=True)
    red.save(red_path)

    green_path = output_dir / "ood" / img_path.with_stem(img_path.stem + "_green")
    green_path.parent.mkdir(parents=True, exist_ok=True)
    green.save(green_path)


cp.create_image_gallery_html(output_dir)
