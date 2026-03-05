import os
import json
import threading
import textworld.gym
from .environment import SingleAlfredTWEnv
from .utils import load_config, process_ob


class ALFWorld_Wrapper:
    def __init__(self, **kwargs):
        # load data_path
        self.data_path = kwargs.get("data_path", None)
        if self.data_path is None:
            raise Exception("missing parameter data_path")
        os.environ["ALFWORLD_DATA"] = self.data_path

        # load config for alfworld benchmark
        self.config_path = kwargs.get("config_path", None)
        if self.config_path is None:
            raise Exception("missing parameter config_path")
        self.config = load_config(self.config_path)

        self._max_id = 0
        self.ls = []
        self.env = {}  # dict[id, env_item]
        self.env_init = {}  # dict[id, env_item]
        self.env_id_map = {}  # dict[id, gym_env_id] for registry cleanup
        self.info = {}  # dict[id, env_info]
        self.games = []  # list[game_file]
        self._lock = threading.Lock()
        
        train_games_root = os.path.join(
            os.environ["ALFWORLD_DATA"], "json_2.1.1", "train"
        )
        test_games_root = os.path.join(
            os.environ["ALFWORLD_DATA"], "json_2.1.1", "valid_train"
        )

        train_mapping_file = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "..",
            "configs",
            "mappings_train.json",
        )
        test_mapping_file = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "..",
            "configs",
            "mappings_test.json",
        )

        with open(train_mapping_file, "r") as f:
            mappings = json.load(f)
            for mapping in mappings:
                self.games.append(
                    os.path.join(
                        train_games_root,
                        mapping["task_type"],
                        mapping["task_id"],
                        "game.tw-pddl",
                    )
                )

        with open(test_mapping_file, "r") as f:
            mappings = json.load(f)
            for mapping in mappings:
                self.games.append(
                    os.path.join(
                        test_games_root,
                        mapping["task_type"],
                        mapping["task_id"],
                        "game.tw-pddl",
                    )
                )

    def create(self):
        try:
            # TODO extend to other kinds of environments
            with self._lock:
                idx = self._max_id
                self._max_id += 1
            self.env[idx] = SingleAlfredTWEnv(self.config)
            self.info[idx] = {"done": False, "reward": 0}
            print(f"-------Env {idx} created--------")
            self.ls.append(idx)
            payload = {"id": idx}
        except Exception as e:
            payload = {"error": f"{e}"}
        return payload
    
    def _cleanup_env(self, idx: int):
        """Close gym env and remove its registry entry to free memory."""
        if idx in self.env_init:
            try:
                self.env_init[idx].close()
            except Exception:
                pass
            del self.env_init[idx]
        # Remove from textworld gym registry
        if idx in self.env_id_map:
            textworld.gym.registry.pop(self.env_id_map[idx], None)
            del self.env_id_map[idx]

    def close(self, idx: int):
        try:
            if idx not in self.info:
                return {"error": f"The id {idx} is not valid."}
            self._cleanup_env(idx)
            if idx in self.env:
                del self.env[idx]
            if idx in self.info:
                del self.info[idx]
            if idx in self.ls:
                self.ls.remove(idx)
            print(f"-------Env {idx} closed--------")
            return {"id": idx, "status": "closed"}
        except Exception as e:
            return {"error": str(e)}

    def __del__(self):
        for idx in list(self.ls):
            self._cleanup_env(idx)
            print(f"-------Env {idx} closed--------")

    def step(self, idx: int, action: str):
        try:
            self._check_id(idx)
            ob, _, done, info = self.env_init[idx].step([action])
            ob, reward, done = process_ob(ob[0]), float(info["won"][0]), done[0]
            available_actions = info.get("admissible_commands", [[]])[0]
            payload = {
                "observation": ob,
                "reward": reward,
                "available_actions": available_actions,
                "done": done,
            }
            self.info[idx].update(payload)
        except Exception as e:
            print("Error id: ", idx)
            payload = {"error": f"{e}"}
        return payload

    def reset(self, idx: int, game: int, world_type: str):
        if world_type not in ["Text", "Embody", "Hybrid"]:
            return {"error": 'world_type must be one of "Text", "Embody" and "Hybrid"'}
        try:
            self._check_id(idx, True)
            # Clean up previous gym env and registry entry
            self._cleanup_env(idx)
            self.env[idx].game_files = [self.games[game]]
            self.env[idx].num_games = 1
            # Track registry keys before/after to capture the new env_id
            keys_before = set(textworld.gym.registry.keys())
            self.env_init[idx] = self.env[idx].init_env(batch_size=1)
            keys_after = set(textworld.gym.registry.keys())
            new_keys = keys_after - keys_before
            if new_keys:
                self.env_id_map[idx] = new_keys.pop()
            ob, info = self.env_init[idx].reset()
            ob = "\n".join(ob[0].split("\n\n")[1:])
            available_actions = info.get("admissible_commands", [[]])[0]
            payload = {
                "id": idx,
                "observation": ob,
                "available_actions": available_actions,
                "task_type": "/".join(info["extra.gamefile"][0].split("/")[-3:-1]),
            }
            self.info[idx] = {
                "world_type": world_type,
                "game": game,
                "observation": ob,
                "available_actions": available_actions,
                "done": False,
                "reward": 0,
            }
        except Exception as e:
            payload = {"error": str(e)}
        return payload

    def get_observation(self, idx: int):
        try:
            self._check_id(idx)
            return self.info[idx]["observation"]
        except Exception as e:
            return {"error": str(e)}

    def get_available_actions(self, idx: int):
        try:
            self._check_id(idx)
            return self.info[idx]["available_actions"]
        except Exception as e:
            return {"error": str(e)}

    def get_detailed_info(self, idx: int):
        try:
            self._check_id(idx)
            return self.info[idx]
        except Exception as e:
            return {"error": str(e)}

    def _check_id(self, idx: int, is_reset: bool = False):
        if idx not in self.info:
            raise NameError(f"The id {idx} is not valid.")
        if not is_reset and self.info[idx]["done"]:
            print("is reset", is_reset)
            print("done", self.info[idx]["done"])
            raise NameError(f"The task with environment {idx} has finished.")


os.environ["ALFWORLD_DATA"] = os.path.expanduser("~/.cache/alfworld")
server = ALFWorld_Wrapper(
    data_path=os.environ["ALFWORLD_DATA"],
    config_path=os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "..", "configs", "base_config.yaml"
    ),
)
