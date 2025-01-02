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
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument(
        "config_file",
        type=types.config_file,
        help="Path to training config file.\n"
        "Download and modify the template from "
        "https://github.com/paninski-lab/lightning-pose/blob/main/scripts/configs/config_default.yaml",
    )
    train_parser.add_argument(
        "--model_dir",
        type=types.model_dir,
        help="The directory under ./outputs/ for the model.\n"
        "  * Defaults to the run timestamp.\n"
        "  * If model_dir is one level deep, model_dir will be model_dir/run_timestamp.\n"
        "  * If model_dir is two levels deep, model_dir will be model_dir without modification.\n",
    )
    train_parser.add_argument(
        "--overrides", nargs="*", help="Config overrides; uses Hydra syntax."
    )

    # Add arguments specific to the 'train' command here

    # Predict command
    predict_parser = subparsers.add_parser("predict")
    predict_parser.add_argument(
        "--model_dir",
        type=types.existing_model_dir,
        help="The directory under ./outputs/ for the model.")
    # Add arguments specific to the 'predict' command here

    # If no commands provided, display the help message.
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()

    if args.command == "train":
        import hydra
        import datetime
        import os

        # Simplified ISO format, e.g. "2024-12-30T19:58:50"
        curr_datetime = datetime.datetime.now().replace(microsecond=0).isoformat()
        model_dir = curr_datetime
        if args.model_dir:
            if len(args.model_dir.parts) == 1:
                model_dir = args.model_dir / curr_datetime
            if len(args.model_dir.parts) == 2:
                model_dir = args.model_dir
        output_dir = Path("outputs") / model_dir

        print(f"Output directory: {output_dir}")
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
