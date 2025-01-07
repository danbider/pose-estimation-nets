"""
litpose generate_config data_dir config_name
litpose train config_name
litpose train <data_dir>

/outputs

"""
import sys
from pathlib import Path
from . import friendly
from . import types


def main():
    parser = friendly.ArgumentParser()
    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
        help="Litpose command to run.",
        parser_class=friendly.ArgumentSubParser,
    )

    # Train command
    train_parser = subparsers.add_parser(
        "train",
        description="Train a lightning-pose model using the specified configuration file.",
        usage="litpose train <config_file> \\\n"
        "                      [--output_dir OUTPUT_DIR] \\\n"
        "                      [--overrides KEY=VALUE...]"
        "",
    )
    train_parser.add_argument(
        "config_file",
        type=types.config_file,
        help="path a config file.\n"
        "Download and modify the config template from: \n"
        "https://github.com/paninski-lab/lightning-pose/blob/main/scripts/configs/config_default.yaml",
    )
    train_parser.add_argument(
        "--output_dir",
        type=types.model_dir,
        help="explicitly specifies the output model directory.\n"
        "If not specified, defaults to "
        "./outputs/{YYYY-MM-DD}/{HH:MM:SS}/",
    )
    train_parser.add_argument(
        "--overrides",
        nargs="*",
        metavar="KEY=VALUE",
        help="overrides attributes of the config file. Uses hydra syntax:\n"
        "https://hydra.cc/docs/advanced/override_grammar/basic/",
    )

    # Add arguments specific to the 'train' command here

    # Predict command
    predict_parser = subparsers.add_parser(
        "predict",
        description="Predict keypoints on one or more images or videos.",
        usage="litpose predict <model_dir> \\\n"
        "                        [--video_file VIDEO_FILE [VIDEO_FILE ...]] \\\n"
        "                        [--videos_dir VIDEOS_DIR [VIDEOS_DIR ...]] \\\n"
        "                        [--image_file IMAGE_FILE [IMAGE_FILE ...]] \\\n"
        "                        [--images_dir IMAGES_DIR [IMAGES_DIR ...]]",
    )
    predict_parser.add_argument(
        "model_dir", type=types.existing_model_dir, help="path to a model directory"
    )

    argg_input_files = predict_parser.add_argument_group('input files')
    argg_input_files.add_argument(
        "--video_file", type=Path, nargs="+", help="video file to predict on"
    )

    argg_input_files.add_argument(
        "--videos_dir",
        nargs="+",
        type=Path,
        help="directory of video files to predict on",
    )

    argg_input_files.add_argument(
        "--image_file", nargs="+", type=Path, help="image file to predict on"
    )

    argg_input_files.add_argument(
        "--images_dir",
        nargs="+",
        type=Path,
        help="directory of image files to predict on",
    )

    argg_post_prediction = predict_parser.add_argument_group('post-prediction')
    argg_post_prediction.add_argument(
        "--skip_viz",
        default=True,
        type=bool,
        help="generate a labeled video with the predicted keypoints",
    )

    # If no commands provided, display the help message.
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()

    if args.command == "train":
        import hydra
        import datetime
        import os

        # E.g. "2024-12-30_19:58:50"
        if args.output_dir:
            output_dir = args.output_dir
        else:
            now = datetime.datetime.now()
            output_dir = (
                Path("outputs") / now.strftime("%Y-%m-%d") / now.strftime("%H:%M:%S")
            )

        print(f"Output directory: {output_dir.absolute()}")
        if args.overrides:
            print(f"Overrides: {args.overrides}")

        # TODO: Move some aspects of directory mgmt to the train function.
        output_dir.mkdir(parents=True, exist_ok=True)

        with hydra.initialize_config_dir(
            version_base="1.1", config_dir=str(args.config_file.parent.absolute())
        ):
            cfg = hydra.compose(
                config_name=args.config_file.stem, overrides=args.overrides
            )

            # Delay this import because it's slow.
            from lightning_pose.train import train

            # Maintain legacy hydra chdir until downstream no longer depends on it.
            os.chdir(output_dir)
            train(cfg)

    elif args.command == "predict":
        # ... your prediction logic ...
        pass


if __name__ == "__main__":
    main()
