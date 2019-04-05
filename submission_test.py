import os
import sys
import json
import argparse
import shutil
import tempfile
import zipfile
from os.path import join as pjoin
import subprocess


def main():
    parser = argparse.ArgumentParser(description="Test submission for the First TextWorld Problem competition.")
    parser.add_argument("submission", help="Submission file (*.zip)")
    parser.add_argument("games_dir", help="Folder containing games to evaluate on (*.ulx + *.json)")

    parser.add_argument("--debug", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as submission_dir:
        # Extract submission files.
        print("Extracting {} ...".format(args.submission))
        with zipfile.ZipFile(args.submission, 'r') as zip_ref:
            zip_ref.extractall(submission_dir)

        if os.path.isdir(submission_dir) and "custom_agent.py" not in os.listdir(submission_dir):
            msg = ("Can't find 'custom_agent.py'. Make sure all your files are places"
                   " at the root of your submission zip file.")
            print(msg, file=sys.stderr)
            sys.exit(1)

        if os.path.isdir(submission_dir) and "metadata" not in os.listdir(submission_dir):
            msg = ("Can't find a 'metadata' file in your submission zip file.")
            print(msg, file=sys.stderr)
            sys.exit(1)

        ingestion_out = pjoin(submission_dir, "stats.json")
        score_out = pjoin(submission_dir, "scores")
        os.mkdir(score_out)
        starting_kit_dir = os.path.dirname(os.path.abspath(__file__))
        ingestion_program = pjoin(starting_kit_dir, "ingestion.py")
        score_program = pjoin(starting_kit_dir, "score.py")

        ingestion_command = "python3 {} {} {} {}".format(ingestion_program, submission_dir, args.games_dir, ingestion_out)
        if args.debug:
            ingestion_command += " --debug"

        print("Running\n{}".format(ingestion_command))
        subprocess.call(ingestion_command.split())

        try:
            # HACK: Making sure the ingestion.py script didn't fail.
            with open(ingestion_out, 'r') as f:
                json.load(f)

            score_command = "python3 {} {} {}".format(score_program, ingestion_out, score_out)
            print("Running\n{}".format(score_command))
            subprocess.call(score_command.split())

        except json.decoder.JSONDecodeError:
            pass


if __name__ == "__main__":
    main()