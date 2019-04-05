import os
import sys
import json
import shutil
import base64
import argparse
from os.path import join as pjoin

import numpy as np

import matplotlib
# According to https://github.com/codalab/codalab-competitions/wiki/User_DetailedResultsPage
matplotlib.use('Agg') # Very important to make it work on Linux.

import matplotlib.pyplot as plt


MAX_HANDICAP = 5
HANDICAP_ADJUSTMENTS = {
    0: 1.00,
    1: 0.85,
    2: 0.77,
    3: 0.73,
    4: 0.65,
    5: 0.50,
}


def get_total_score(stats):
    score = 0
    for gamefile in stats:
        for no_episode in range(len(stats[gamefile]["runs"])):
            score += stats[gamefile]["runs"][no_episode]["score"]

    return score


def get_max_score(stats):
    max_score = 0
    for gamefile in stats:
        for no_episode in range(len(stats[gamefile]["runs"])):
            max_score += stats[gamefile]["max_scores"]

    return max_score


def get_total_steps(stats):
    steps = 0
    for gamefile in stats:
        for no_episode in range(len(stats[gamefile]["runs"])):
            steps += stats[gamefile]["runs"][no_episode]["steps"]

    return steps


def get_handicap(requested_infos):
    requested_infos = set(requested_infos)
    handicap = 0

    if len(requested_infos & {"description", "inventory"}) > 0:
        handicap = 1

    if len(requested_infos & {"verbs", "command_templates"}) > 0:
        handicap = 2

    if len(requested_infos & {"entities"}) > 0:
        handicap = 3

    if len(requested_infos & {"recipe"}) > 0:
        handicap = 4

    if len(requested_infos & {"admissible_commands"}) > 0:
        handicap = 5

    return handicap


def score_leaderboard(stats, output_dir):
    # Get agent's handicap.
    handicap = get_handicap(stats["requested_infos"])

    # Extract result from stats.
    leaderboard = {}
    leaderboard["score"] = get_total_score(stats["games"])
    leaderboard["max_score"] = get_max_score(stats["games"])
    leaderboard["adjusted_score"] = HANDICAP_ADJUSTMENTS[handicap] * leaderboard["score"]
    leaderboard["nb_steps"] = get_total_steps(stats["games"])
    leaderboard["handicap"] = get_handicap(stats["requested_infos"])

    # Write leaderboard results.
    if not os.path.exists(output_dir):
	    os.makedirs(output_dir)

    content = "\n".join("{}: {}".format(k, v) for k, v in leaderboard.items())
    with open(pjoin(output_dir, "scores.txt"), "w") as f:
        f.write(content)

    stats["leaderboard"] = leaderboard
    print(content)


def score_html(stats, output_dir):
    # Create folders for detailed results.
    html_dir = pjoin(output_dir, "html")
    if not os.path.exists(html_dir):
	    os.makedirs(html_dir)

    images_dir = pjoin(html_dir, "images")
    if not os.path.exists(images_dir):
	    os.makedirs(images_dir)

    #html = "Available during the validation phase."
    # Load template files.
    with open(pjoin(os.path.dirname(__file__), "template.html")) as f:
        html = f.read()
    with open(pjoin(os.path.dirname(__file__), "template.css")) as f:
        css = f.read()
    with open(pjoin(os.path.dirname(__file__), "template.js")) as f:
        js = f.read()
    with open(pjoin(os.path.dirname(__file__), "template_game.html")) as f:
        game_html_template = f.read()

    # Get the information.
    # Agent's handicap.
    handicap = get_handicap(stats["requested_infos"])

    total_score = get_total_score(stats["games"])
    max_score = get_max_score(stats["games"])
    adjusted_score = HANDICAP_ADJUSTMENTS[handicap] * total_score
    nb_steps = get_total_steps(stats["games"])

    # Sort commands according to the number of skills present in the game.
    gamefiles = list(stats["games"])
    gamefiles = sorted(gamefiles)
    gamefiles = sorted(gamefiles, key=lambda e: len(e.split("+")))

    game_html = ""
    for gamefile in gamefiles:
        nb_runs = len(stats["games"][gamefile]["runs"])
        scores = []
        steps = []
        for no_episode in range(nb_runs):
            scores.append(stats["games"][gamefile]["runs"][no_episode]["score"])
            steps.append(stats["games"][gamefile]["runs"][no_episode]["steps"])

        game_score = sum(scores)
        game_score_avg = game_score / nb_runs
        game_max_score = stats["games"][gamefile]["max_scores"] * nb_runs
        game_score_ratio = game_score / game_max_score

        game_name = gamefile[:-4]
        skillset = game_name.rsplit("-", 1)[0].split("-", 2)[-1]
        plot1_path = pjoin(images_dir, game_name + "_plot1.png")
        plot2_path = pjoin(images_dir, game_name + "_plot2.png")
        plot12_path = pjoin(images_dir, game_name + "_plot12.png")

        # Build plot #12
        fig, ax1 = plt.subplots()
        lines = []
        lines += ax1.plot(range(1, nb_runs + 1), scores, color="#1f77b4", label="Score")
        ax1.set_xlabel("Nb. runs")
        ax1.set_ylabel("Score")
        ax1.tick_params(axis='y', labelcolor="#1f77b4")

        ax2 = ax1.twinx()
        lines += ax2.plot(range(1, nb_runs + 1), steps, color="#ff7f0e", label="Moves")
        ax2.set_ylabel("Moves")
        ax2.tick_params(axis='y', labelcolor="#ff7f0e")
        ax1.legend(lines, [l.get_label() for l in lines])
        plt.savefig(plot12_path)
        plt.close()

        plot12_src = pjoin(os.path.basename(images_dir), game_name + "_plot12.png")  # Path relative to ./html/

        # According to https://github.com/codalab/codalab-competitions/wiki/User_DetailedResultsPage
        # encode the image as base64 to embed it in the html.
        with open(plot12_path, 'rb') as f:
            data_uri = base64.b64encode(f.read()).decode().replace('\n', '')
        plot12_src = 'data:image/png;base64,{0}'.format(data_uri)

        status = "low"
        if game_score_ratio >= 0.1:
            status = "mid"
        if game_score_ratio >= 0.9:
            status = "high"

        # Fill template.
        game_html += game_html_template.format(game_name="({:05.1%})\t".format(game_score_ratio) + skillset,
                                               status=status,
                                               game_score=game_score, game_max_score=game_max_score,
                                               game_score_ratio=game_score_ratio,
                                               #plot1_src=plot1_src, plot2_src=plot2_src,
                                               plot12_src=plot12_src,
                                               )

    # Fill template.
    html = html.format(css=css, js=js,
                       total_score=total_score, max_score=max_score, score_ratio=total_score/max_score,
                       adjusted_score=adjusted_score, handicap=handicap,
                       game_stats=game_html)

    # Write detailed results.
    with open(pjoin(html_dir, "detailed_results.html"), "w") as f:
        f.write(html)

def main():
    parser = argparse.ArgumentParser(description="Extract score from `stats.json`.")
    parser.add_argument("stats", help="JSON file")
    parser.add_argument("output_dir")
    parser.add_argument("--html", action="store_true")
    args = parser.parse_args()

    try:
        with open(args.stats) as f:
            stats = json.load(f)
    except (json.decoder.JSONDecodeError, OSError, FileNotFoundError):
        msg = ("Cannot find a valid output for the ingestion program. Most certainly it has failed."
               "\n\nNB: multiple instances of your agent are being run in parallel."
               " If your submission is using the GPU, make sure the total memory usage fits within 16GB."
               " You can control the number of instances by specifying a value for `nb_processes` in your"
               " metadata file (see sample_submission_lstm-dqn.zip)."
               "\n\nMore information might be available in the ingestion error log.")
        raise NameError(msg)

    score_leaderboard(stats, args.output_dir)
    if args.html:
        score_html(stats, args.output_dir)


if __name__ == "__main__":
    main()
