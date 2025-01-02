import subprocess

def run_command(command):
  """Helper function to run a command and return its output."""
  result = subprocess.run(command, shell=True, capture_output=True, text=True)
  return result.stdout.strip(), result.stderr.strip(), result.returncode

def test_litpose_help_message():
  """Test a basic command with no arguments."""
  output, error, returncode = run_command("litpose")
  assert returncode == 1  # Expect successful execution
  assert output == """Welcome to the lightning-pose CLI!

usage: litpose [-h] {train,predict} ...

positional arguments:
  {train,predict}  Litpose command to run.

options:
  -h, --help       show this help message and exit

documentation:
https://lightning-pose.readthedocs.io/en/latest/source/user_guide/index.html"""
  assert error == ""  # Expect no errors

def test_litpose_train_help_message():
  """Test a command with arguments."""
  output, error, returncode = run_command("litpose train")
  assert returncode == 2
  assert output == """usage: litpose train [-h] [--model_dir MODEL_DIR]
                     [--overrides [OVERRIDES ...]]
                     config_file

positional arguments:
  config_file           Path to training config file.
                        Download and modify the template from
                        https://github.com/paninski-lab/lightning-pose/blob/main/scripts/configs/config_default.yaml

options:
  -h, --help            show this help message and exit
  --model_dir MODEL_DIR
                        The directory under ./outputs/ for the model.
                          * Defaults to the run timestamp.
                          * If model_dir is one level deep, model_dir will be
                        model_dir/run_timestamp.
                          * If model_dir is two levels deep, model_dir will be
                        model_dir without modification.
                        
  --overrides [OVERRIDES ...]
                        Config overrides; uses Hydra syntax.

documentation:
https://lightning-pose.readthedocs.io/en/latest/source/user_guide/index.html"""

  assert error == """\x1b[91merror:
the following arguments are required: config_file

\x1b[0m--------------------------------------------------------------------------------"""
