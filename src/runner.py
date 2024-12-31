import subprocess
import json
import sys

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# # Load common arguments
# with open("common_args.json") as f:
#     common_args = json.load(f)

# # Load script-specific arguments and merge with common arguments


# for script, arg_file in [
#     ("script1.py", "script1_args.json"),
#     ("script2.py", "script2_args.json"),
#     ("script3.py", "script3_args.json"),
# ]:
#     with open(arg_file) as f:
#         script_args = json.load(f)
#     subprocess.run(["python", script, *common_args, *script_args])


def main():
    logger.info("Entering runner script")
    files_to_run = [
        "src/data_prep.py",
        "src/train.py",
        "src/predict.py",
    ]

    common_json = "config/common.json"
    with open(common_json) as f:
        common_args = json.load(f)

    for file in files_to_run:
        logger.info(f"Running {file}")
        file_name = file.split("/")[-1].split(".")[0]
        json_file = f"config/{file_name}.json"

        with open(json_file) as f:
            script_args = json.load(f)

        all_args = common_args | script_args

        json_text = json.dumps(all_args)

        subprocess.run([sys.executable, file, f"--json_args={json_text}"])


if __name__ == "__main__":
    main()
