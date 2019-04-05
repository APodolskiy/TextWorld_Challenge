#!/usr/bin/env python3

import argparse
import docker
import flask
import functools
import glob
import gym
import json
import multiprocessing
import os
import requests
import subprocess
import sys
import tempfile
import textworld
import textworld.gym
import threading
import time
import tqdm
import urllib


NB_EPISODES = 10
MAX_EPISODE_STEPS = 100
TIMEOUT = 6 * 3600  # 6 hours
PORT = 29592
DEFAULT_IMAGE = "tavianator/textworld-codalab"

# List of additional information available during evaluation.
AVAILABLE_INFORMATION = textworld.EnvInfos(
    max_score=True, has_won=True, has_lost=True,                    # Handicap 0
    description=True, inventory=True, objective=True,               # Handicap 1
    verbs=True, command_templates=True,                             # Handicap 2
    entities=True,                                                  # Handicap 3
    extras=["recipe"],                                              # Handicap 4
    admissible_commands=True,                                       # Handicap 5
)


def _validate_requested_infos(infos):
    msg = "The following information cannot be requested: {}"
    for key in infos.basics:
        if not getattr(AVAILABLE_INFORMATION, key):
            raise ValueError(msg.format(key))

    for key in infos.extras:
        if key not in AVAILABLE_INFORMATION.extras:
            raise ValueError(msg.format(key))


def _serialize_infos(infos):
    result = {}
    for slot in infos.__slots__:
        result[slot] = getattr(infos, slot)
    return result


class _RemoteAgent:
    """
    An agent that forwards to a remote agent in another process.
    """

    def __init__(self, host="localhost:{}".format(PORT)):
        self._host = host

    def _send(self, command, *args):
        url = "http://{}/agents/{}/{}".format(self._host, self._game_id, command)
        args = json.dumps(args).encode("utf-8")
        request = urllib.request.Request(url, args)
        try:
            response = urllib.request.urlopen(request)
            result = response.read().decode("utf-8")
        except urllib.error.HTTPError:
            raise ValueError("Some errors occurred when evaluating the agent. You can test your agent"
                " using the `test_submission.py` script provided with the starting kit"
                " and using the `--debug` flag."
                " If you can't find your error, reach out to us: textworld@microsoft.com.") \
                from None

        return json.loads(result)

    def train(self):
        self._send("train")

    def eval(self):
        self._send("eval")

    def select_additional_infos(self):
        return textworld.EnvInfos(**self._send("select_additional_infos"))

    def act(self, obs, scores, dones, infos):
        return self._send("act", obs, scores, dones, infos)

    def close(self):
        return self._send("stop")


class _AgentProcess:
    """
    An agent that lives in a remote process.
    """

    def __init__(self, agent_class):
        # We rely on getting EOFError from pipes for the remote agent, so we
        # can't have multiprocessing use the fork start method, since the forked
        # processes will inherit the pipes without closing them
        ctx = multiprocessing.get_context("forkserver")

        self._parent_conn, self._child_conn = ctx.Pipe()

        args = (agent_class, self._child_conn, self._parent_conn)
        self._process = ctx.Process(target=self._run, args=args)

        self._process.start()
        self._child_conn.close()

    @staticmethod
    def _run(agent_class, conn, parent_conn):
        parent_conn.close()

        agent = agent_class()
        try:
            while True:
                try:
                    command, args = conn.recv()
                except EOFError:
                    break
                result = getattr(agent, command)(*args)
                conn.send(result)
        finally:
            conn.close()

    def call(self, command, args):
        self._parent_conn.send((command, args))
        return self._parent_conn.recv()

    def stop(self):
        self._parent_conn.close()
        self._process.join()


def _serve(agent_class, args):
    """
    Start a web server that delegates requests to the custom agent.
    """

    app = flask.Flask("TextWorld")

    agents = {}
    lock = threading.Lock()

    def _get_agent(game_id):
        with lock:
            agent = agents.get(game_id)
            if agent is None:
                agent = _AgentProcess(agent_class)
                agents[game_id] = agent
            return agent

    @app.route("/agents/<game_id>/<command>", methods=["POST"])
    def _call(game_id, command):
        args = flask.request.get_json(force=True)
        result = _get_agent(game_id).call(command, args)
        if command == "select_additional_infos":
            result = _serialize_infos(result)
        return json.dumps(result)

    @app.route("/agents/<game_id>/stop", methods=["POST"])
    def _stop_agent(game_id):
        with lock:
            agent = agents.pop(game_id, None)
        if agent is not None:
            agent.stop()
        return json.dumps(None)

    @app.route("/stop")
    def _stop():
        flask.request.environ.get("werkzeug.server.shutdown")()
        return ""

    app.run(host="0.0.0.0", port=args.listen, threaded=True)


def _play_game(agent_class, gamefile):
    game_name = os.path.basename(gamefile)

    agent = agent_class()
    if isinstance(agent, _RemoteAgent):
        # HACK: tell the remote agent the game ID so it can talk to the right remote agent
        agent._game_id = game_name

    agent.eval()
    requested_infos = agent.select_additional_infos()
    _validate_requested_infos(requested_infos)

    # Turn on flags needed for the evaluation.
    requested_infos.has_won = True
    requested_infos.has_lost = True
    requested_infos.max_score = True

    stats = {}
    start_time = time.time()

    stats["runs"] = []

    name = "test_{}".format(hash(gamefile))
    env_id = textworld.gym.register_games([gamefile], requested_infos,
                                            max_episode_steps=MAX_EPISODE_STEPS,
                                            name=name)
    env_id = textworld.gym.make_batch(env_id, batch_size=1)
    env = gym.make(env_id)

    for no_episode in range(NB_EPISODES):
        obs, infos = env.reset()

        all_commands = []
        scores = [0] * len(obs)
        dones = [False] * len(obs)
        steps = [0] * len(obs)
        while not all(dones):
            # Increase step counts.
            steps = [step + int(not done) for step, done in zip(steps, dones)]

            commands = agent.act(obs, scores, dones, infos)
            all_commands.append(commands)
            obs, scores, dones, infos = env.step(commands)

        # Let the agent knows the game is done.
        agent.act(obs, scores, dones, infos)

        # Collect stats
        stats["runs"].append({})
        stats["runs"][no_episode]["score"] = scores[0]
        stats["runs"][no_episode]["steps"] = steps[0]
        stats["runs"][no_episode]["commands"] = [cmds[0] for cmds in all_commands]
        stats["runs"][no_episode]["has_won"] = infos["has_won"][0]
        stats["runs"][no_episode]["has_lost"] = infos["has_lost"][0]

    env.close()
    if hasattr(agent, "close"):
        agent.close()

    stats["max_scores"] = infos["max_score"][0]
    elapsed = time.time() - start_time
    stats["duration"] = elapsed

    return {game_name: stats}, requested_infos.basics + requested_infos.extras


def _evaluate(agent_class, game_files, nb_processes):
    stats = {"games": {}, "requested_infos": [], "game_files": game_files}

    print("Using {} processes.".format(nb_processes))
    desc = "Evaluating {} games".format(len(game_files))
    pbar = tqdm.tqdm(total=len(game_files), desc=desc)

    def _assemble_results(args):
        data, requested_infos = args
        stats["games"].update(data)
        stats["requested_infos"] = requested_infos

        game_name, infos = list(data.items())[0]
        total_scores = sum(d["score"] for d in infos["runs"])
        total_steps = sum(d["steps"] for d in infos["runs"])

        desc = "{:2d} / {}:\t{}".format(total_scores, total_steps, game_name)
        pbar.write(desc)
        pbar.update()

    if nb_processes > 1:
        pool = multiprocessing.Pool(nb_processes)
        results = []
        for game_file in game_files:
            result = pool.apply_async(_play_game, (agent_class, game_file), callback=_assemble_results)
            results.append(result)

        for result in results:
            result.get()

        pool.close()
        pool.join()
        pbar.close()

    else:
        for game_file in game_files:
            data = _play_game(agent_class, game_file)
            _assemble_results(data)

        pbar.close()

    return stats


def _run_evaluation(agent_class, args):
    games = glob.glob(os.path.join(args.games_dir, "**/*.ulx"), recursive=True)
    stats = _evaluate(agent_class, games, args.nb_processes)

    out_dir = os.path.dirname(os.path.abspath(args.output))
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    with open(args.output, "w") as f:
        json.dump(stats, f)


def _dockerize_agent(args, client, image, runtime):
    submission_dir = os.path.abspath(args.submission_dir)
    self_file = os.path.abspath(__file__)

    volumes = {
        submission_dir: {
            "bind": "/usr/src/submission",
            "mode": "ro",
        },
        self_file: {
            "bind": "/usr/bin/ingestion.py",
            "mode": "ro",
        },
    }

    environment = {
        "PYTHONUNBUFFERED": "1",
        "MKL_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
    }

    command = [
        "python3",
        "/usr/bin/ingestion.py",
        "--in-docker",
        "--listen={}".format(PORT),
        "/usr/src/submission",
        "/dev/null",
    ]

    if args.debug:
        command += ["--debug"]

    if args.nb_processes:
        command += ["--nb-processes", str(args.nb_processes)]

    return client.containers.run(
        image,
        command,
        runtime=runtime,
        detach=True,
        network="textworld",
        volumes=volumes,
        environment=environment,
    )


def _dockerize_evaluator(args, client, host, runtime):
    games_dir = os.path.abspath(args.games_dir)
    output_file = os.path.abspath(args.output)
    self_file = os.path.abspath(__file__)

    volumes = {
        games_dir: {
            "bind": "/usr/share/textworld-games",
            "mode": "ro",
        },
        output_file: {
            "bind": "/usr/share/textworld-stats.json",
            "mode": "rw",
        },
        self_file: {
            "bind": "/usr/bin/ingestion.py",
            "mode": "ro",
        },
    }

    environment = {
        "PYTHONUNBUFFERED": "1",
    }

    command = [
        "python3",
        "/usr/bin/ingestion.py",
        "--in-docker",
        "--remote={}".format(host),
        "/dev/null",
        "/usr/share/textworld-games",
        "/usr/share/textworld-stats.json",
    ]

    if args.debug:
        command += ["--debug"]

    if args.nb_processes:
        command += ["--nb-processes", str(args.nb_processes)]

    return client.containers.run(
        DEFAULT_IMAGE,
        command,
        runtime=runtime,
        detach=True,
        network="textworld",
        volumes=volumes,
        environment=environment,
    )


def _dockerize(args):
    client = docker.from_env()

    info = client.info()
    if "nvidia" in info["Runtimes"]:
        runtime = "nvidia"
    else:
        runtime = info["DefaultRuntime"]

    # If it doesn't exist already, create a docker network with no internet access
    try:
        client.networks.create("textworld", internal=True, check_duplicate=True)
    except docker.errors.APIError as e:
        # HTTP 409: Conflict, aka the network already existed
        if e.status_code != 409:
            raise

    image_path = os.path.join(args.submission_dir, "Dockerimage")
    if os.path.exists(image_path):
        with open(image_path, "r") as f:
            image = f.read().strip()
    else:
        image = DEFAULT_IMAGE

    # Make sure the stats file exists so Docker doesn't create it as a directory
    open(args.output, "w").close()

    print("Loading {}...".format(image))
    agent = _dockerize_agent(args, client, image, runtime)
    evaluator = None

    try:
        agent.reload()
        host = agent.attrs["NetworkSettings"]["Networks"]["textworld"]["IPAddress"]
        host = "{}:{}".format(host, PORT)

        # HACK: Need to wait until the container web server is up
        time.sleep(10)
        print("Loading {}...".format(DEFAULT_IMAGE))
        evaluator = _dockerize_evaluator(args, client, host, runtime)
        print("Running {}...".format(image))
        for log in evaluator.logs(stream=True, follow=True):
            sys.stdout.buffer.write(log)
    finally:
        if evaluator:
            evaluator.stop()
        agent.stop()

        if args.verbose:
            sys.stdout.buffer.write(agent.logs(stdout=True, stderr=False))
            sys.stderr.buffer.write(agent.logs(stdout=False, stderr=True))

        if evaluator:
            evaluator.remove(force=True)
        agent.remove(force=True)

    print("Done")


def main():
    parser = argparse.ArgumentParser(description="Evaluate an agent.")
    parser.add_argument("--in-docker", action="store_true", default=False, help=argparse.SUPPRESS)
    parser.add_argument("--listen", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--remote", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("submission_dir")
    parser.add_argument("games_dir")
    parser.add_argument("output", nargs='?', default="stats.json")
    parser.add_argument("--nb-processes", type=int)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    if os.path.isdir(args.submission_dir) and "custom_agent.py" not in os.listdir(args.submission_dir):
        msg = ("Can't find 'custom_agent.py'. Make sure all your files are places"
               " at the root of your submission zip file.")
        parser.error(msg)

    metadata = {}
    submission_metadata = os.path.join(args.submission_dir, "metadata")
    if os.path.isdir(args.submission_dir):
        if not os.path.isfile(submission_metadata):
            msg = ("Can't find a 'metadata' file in your submission zip file.")
            parser.error(msg)

        with open(submission_metadata) as f:
            for line in f:
                key, value = line.split(":", 1)
                metadata[key.strip()] = value.strip()

    args.nb_processes = args.nb_processes or metadata.get("nb_processes") or multiprocessing.cpu_count()
    args.nb_processes = int(args.nb_processes)
    if args.debug:
        args.nb_processes = 1
        args.verbose = True

    if args.in_docker:
        args.submission_dir = os.path.abspath(args.submission_dir)
        args.games_dir = os.path.abspath(args.games_dir)
        args.output = os.path.abspath(args.output)

        if args.listen is not None:
            os.chdir(args.submission_dir)  # Needed to load local files (e.g. vocab.txt)
            sys.path = [args.submission_dir] + sys.path  # Prepend to PYTHONPATH
            from custom_agent import CustomAgent
            _serve(CustomAgent, args)
        elif args.remote is not None:
            _run_evaluation(functools.partial(_RemoteAgent, host=args.remote), args)
        else:
            os.chdir(args.submission_dir)  # Needed to load local files (e.g. vocab.txt)
            sys.path = [args.submission_dir] + sys.path  # Prepend to PYTHONPATH
            from custom_agent import CustomAgent
            _run_evaluation(CustomAgent, args)
    else:
        _dockerize(args)

if __name__ == "__main__":
    main()
