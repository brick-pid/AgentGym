from scienceworld import ScienceWorldEnv

from agentenv_pool import BaseEnvWrapper
from agentenv_pool.errors import (
    EnvNotFoundError,
    EnvClosedError,
    EpisodeFinishedError,
)


class SciWorldWrapper(BaseEnvWrapper):
    def __init__(self):
        self._max_id = 0
        self.env = {}
        self.info = {}
        self.games = []
        self.ls = []
        exceptions = {"5-1", "5-2", "9-1", "9-2", "9-3", "10-1", "10-2"}
        init_env = ScienceWorldEnv()
        for key, value in init_env.tasks.items():
            if key not in exceptions:
                self.games += [
                    {"taskName": value, "variationIdx": i}
                    for i in range(init_env.get_max_variations(value))
                ]
        init_env.close()
        del init_env

    def create_with_id(self, env_id: int):
        env = ScienceWorldEnv()
        self.env[env_id] = env
        self.info[env_id] = {"deleted": False, "done": False}
        self.ls.append(env_id)
        print(f"-------Env {env_id} created--------")
        return {"env_id": env_id}

    def step(self, env_id: int, action: str):
        self._check_env_id(env_id)
        if "task_name" not in self.info[env_id]:
            raise EpisodeFinishedError(
                f"Environment {env_id} has not been reset. "
                "Please call reset before step."
            )
        ob, reward, done, info = self.env[env_id].step(action)
        payload = {
            "observation": ob,
            "reward": reward,
            "score": info["score"],
            "done": done,
        }
        self.info[env_id].update(payload)
        return payload

    def step_visual(self, env_id: int, action: str):
        self._check_env_id(env_id)
        processed_action = action
        if processed_action.endswith("</s>"):
            processed_action = processed_action[:-4]
        if "Action:" in processed_action:
            action_parts = processed_action.split("Action:")
            if len(action_parts) > 1:
                processed_action = action_parts[1].strip()
            else:
                processed_action = action_parts[0].strip()
        ob, reward, done, info = self.env[env_id].step(processed_action)
        try:
            object_tree = self.env[env_id].get_object_tree()
        except Exception:
            object_tree = None
        try:
            inventory = self.env[env_id].inventory()
        except Exception:
            inventory = ""
        payload = {
            "observation": ob,
            "reward": reward,
            "score": info["score"],
            "done": done,
            "info": info,
            "object_tree": object_tree,
            "inventory": inventory,
            "moves": info.get("moves", 0),
        }
        self.info[env_id].update(payload)
        return payload

    def reset(self, env_id: int, task_id=None):
        if task_id is None:
            task_id = 0
        self._check_env_id(env_id, True)
        self.env[env_id].load(
            self.games[task_id]["taskName"],
            self.games[task_id]["variationIdx"],
        )
        task_description = self.env[env_id].get_task_description()
        ob, reward, done, info = self.env[env_id].step("look around")
        payload = {
            "task_name": self.games[task_id]["taskName"],
            "var_num": self.games[task_id]["variationIdx"],
            "task_description": task_description,
            "observation": ob,
            "reward": reward,
            "score": info["score"],
            "deleted": False,
            "done": done,
        }
        self.info[env_id].update(payload)
        return payload

    def get_observation(self, env_id: int):
        self._check_env_id(env_id)
        return self.info[env_id]["observation"]

    def get_action_hint(self, env_id: int):
        self._check_env_id(env_id)
        return {
            "possible_actions": self.env[env_id].get_possible_actions(),
            "possible_objects": self.env[env_id].get_possible_objects(),
        }

    def get_goals(self, env_id: int):
        self._check_env_id(env_id)
        return {"goals": self.env[env_id].get_goal_progress_str()}

    def get_detailed_info(self, env_id: int):
        self._check_env_id(env_id)
        return self.info[env_id]

    def _check_env_id(self, env_id: int, is_reset: bool = False):
        if env_id not in self.info:
            raise EnvNotFoundError(f"The id {env_id} is not valid.")
        if self.info[env_id]["deleted"]:
            raise EnvClosedError(
                f"The task with environment {env_id} has been deleted."
            )
        if not is_reset and self.info[env_id]["done"]:
            raise EpisodeFinishedError(
                f"The task with environment {env_id} has finished."
            )

    def close(self, env_id: int):
        if env_id not in self.info:
            raise EnvNotFoundError(f"The id {env_id} is not valid.")
        if self.info[env_id]["deleted"]:
            raise EnvClosedError(
                f"The task with environment {env_id} has been deleted."
            )
        self.env[env_id].close()
        self.info[env_id]["deleted"] = True
        self.ls.remove(env_id)
        print(f"-------Env {env_id} closed--------")
        return True

    def get_task_description(self, env_id: int):
        self._check_env_id(env_id)
        task_desc = self.env[env_id].get_task_description()
        return {"task_description": task_desc}

    def get_object_tree(self, env_id: int):
        self._check_env_id(env_id)
        object_tree = self.env[env_id].get_object_tree()
        return {"object_tree": object_tree}

    def get_current_state(self, env_id: int):
        self._check_env_id(env_id)
        state = {
            "observation": self.env[env_id].look(),
            "inventory": self.env[env_id].inventory(),
            "task_description": self.env[env_id].get_task_description(),
            "goal_progress": self.env[env_id].get_goal_progress(),
            "possible_actions": self.env[env_id].get_possible_actions()[:10],
            "possible_objects": self.env[env_id].get_possible_objects()[:10],
            "current_moves": self.env[env_id].get_num_moves(),
            "environment_info": self.info[env_id],
        }
        return state
