from typing import List
from legent import Action, Observation
from legent.utils.math import vec_xz, distance
import numpy as np


class Predicate:
    def __init__(self) -> None:
        pass

    def task_done(self, action: Action, obs: Observation, options, task_setting) -> int:
        # 0 continue 1 success -1 failed
        raise NotImplementedError


class PredicateChoose(Predicate):
    def __init__(self, answer) -> None:
        self.answer = answer

    def task_done(self, action: Action, obs: Observation, options, task_setting) -> int:
        if options[action.action_choice] == self.answer:
            return 1, {}
        elif options[action.action_choice].startswith("answer"):
            return -1, {}
        else:
            return 0, {}


class PredicateAgentNear(Predicate):
    def __init__(self, object_id) -> None:
        self.object_id = object_id

    def task_done(self, action: Action, obs: Observation, options, task_setting) -> int:

        game_states = obs.game_states
        d = distance(vec_xz(game_states["agent"]["position"]), vec_xz(game_states["instances"][self.object_id]["position"]))
        if d < 1.5:
            return 1, {"distance": d}
        else:
            return 0, {"distance": d}


class PredicateCloser(Predicate):
    def __init__(self, object1, object2, obs: Observation) -> None:
        self.object1 = object1
        self.object2 = object2
        instances = obs.game_states["instances"]
        self.init_distance = distance(vec_xz(instances[object1]["position"]), vec_xz(instances[object2]["position"]))

    def task_done(self, action: Action, obs: Observation, options, task_setting) -> int:
        instances = obs.game_states["instances"]
        d = distance(vec_xz(instances[self.object1]["position"]), vec_xz(instances[self.object2]["position"]))
        if self.init_distance - d > 2:
            return 1, {"distance_closer": self.init_distance - d}
        else:
            return 0, {"distance_closer": self.init_distance - d}


class PredicateFurther(Predicate):
    def __init__(self, object1, object2, obs: Observation) -> None:
        self.object1 = object1
        self.object2 = object2
        instances = obs.game_states["instances"]
        self.init_distance = distance(vec_xz(instances[object1]["position"]), vec_xz(instances[object2]["position"]))

    def task_done(self, action: Action, obs: Observation, options, task_setting) -> int:
        instances = obs.game_states["instances"]
        d = distance(vec_xz(instances[self.object1]["position"]), vec_xz(instances[self.object2]["position"]))
        if d - self.init_distance > 1:
            return 1, {"distance_closer": d - self.init_distance}
        else:
            return 0, {"distance_closer": d - self.init_distance}


class PredicateOn(Predicate):
    def __init__(self, object1, object2) -> None:
        self.object1 = object1
        self.object2 = object2

    def task_done(self, action: Action, obs: Observation, options, task_setting) -> int:
        instances = obs.game_states["instances"]
        d = distance(vec_xz(instances[self.object1]["position"]), vec_xz(instances[self.object2]["position"]))
        if d < 0.3:
            return 1, {}
        else:
            return 0, {}


class PredicateGrab(Predicate):
    def __init__(self, object) -> None:
        self.object = object

    def task_done(self, action: Action, obs: Observation, options, task_setting) -> int:
        done = obs.game_states["agent_grab_instance"] == self.object
        if done:
            return 1, {}
        else:
            return 0, {}

class PredicateGrabOnce(Predicate):
    def __init__(self, object) -> None:
        self.object = object
        self.grabbed_once = False

    def task_done(self, action: Action, obs: Observation, options, task_setting) -> int:
        if self.grabbed_once:
            return 1, {}
        
        done = obs.game_states["agent_grab_instance"] == self.object
        if done:
            self.grabbed_once = True
            return 1, {}
        else:
            return 0, {}

class PredicateSwap(Predicate):
    def __init__(self, object1, object2, obs: Observation) -> None:
        self.object1 = object1
        self.object2 = object2
        instances = obs.game_states["instances"]
        self.init_pos1 = instances[object1]["position"]
        self.init_pos2 = instances[object2]["position"]

    def task_done(self, action: Action, obs: Observation, options, task_setting) -> int:
        instances = obs.game_states["instances"]
        pos1 = instances[self.object1]["position"]
        pos2 = instances[self.object2]["position"]

        if distance(vec_xz(self.init_pos1), vec_xz(pos2)) < 1 and distance(vec_xz(self.init_pos2), vec_xz(pos1)) < 1:
            return 1, {}
        else:
            return 0, {}


class PredicateIn(Predicate):
    def __init__(self, objects, on_object) -> None:
        self.objects = objects
        self.on_object = on_object

    def task_done(self, action: Action, obs: Observation, options, task_setting) -> int:
        instances = obs.game_states["instances"]
        for object_id in self.objects:
            d = distance(vec_xz(instances[object_id]["position"]), vec_xz(instances[self.on_object]["position"]))
            if d > 0.3:
                return 0, {}

        return 1, {}


class PredicateNotOn(Predicate):
    def __init__(self, objects, on_object) -> None:
        self.objects = objects
        self.on_object = on_object

    def task_done(self, action: Action, obs: Observation, options, task_setting) -> int:
        instances = obs.game_states["instances"]
        for object_id in self.objects:
            d = distance(vec_xz(instances[object_id]["position"]), vec_xz(instances[self.on_object]["position"]))
            if d < 1:
                return 0, {}

        return 1, {}


# TODO: near和at基本一样，可以合并
class PredicateAgentAt(Predicate):
    def __init__(self, object_id) -> None:
        self.object_id = object_id

    def task_done(self, action: Action, obs: Observation, options, task_setting) -> int:

        game_states = obs.game_states
        d = distance(vec_xz(game_states["agent"]["position"]), vec_xz(game_states["instances"][self.object_id]["position"]))
        if d < 0.3:
            return 1, {"distance": d}
        else:
            return 0, {"distance": d}


class PredicateAt(Predicate):
    def __init__(self, object1, object2) -> None:
        self.object1 = object1
        self.object2 = object2

    def task_done(self, action: Action, obs: Observation, options, task_setting) -> int:
        instances = obs.game_states["instances"]
        d = distance(vec_xz(instances[self.object1]["position"]), vec_xz(instances[self.object2]["position"]))
        if d < 0.3:
            return 1, {"distance": d}
        else:
            return 0, {"distance": d}


class PredicateNear(Predicate):
    def __init__(self, object1, object2) -> None:
        self.object1 = object1
        self.object2 = object2

    def task_done(self, action: Action, obs: Observation, options, task_setting) -> int:
        instances = obs.game_states["instances"]
        d = distance(vec_xz(instances[self.object1]["position"]), vec_xz(instances[self.object2]["position"]))
        if d < 1.5:
            return 1, {"distance": d}
        else:
            return 0, {"distance": d}


class PredicateNotNear(Predicate):
    def __init__(self, object1, object2) -> None:
        self.object1 = object1
        self.object2 = object2

    def task_done(self, action: Action, obs: Observation, options, task_setting) -> int:
        instances = obs.game_states["instances"]
        d = distance(vec_xz(instances[self.object1]["position"]), vec_xz(instances[self.object2]["position"]))
        if d > 1.5:
            return 1, {"distance": d}
        else:
            return 0, {"distance": d}


class PredicateNotAt(Predicate):
    def __init__(self, object1, object2) -> None:
        self.object1 = object1
        self.object2 = object2

    def task_done(self, action: Action, obs: Observation, options, task_setting) -> int:
        instances = obs.game_states["instances"]

        def vec_xyz(position):
            return np.array([position["x"], position["y"], position["z"]])

        d = np.linalg.norm(vec_xyz(instances[self.object1]["position"]) - vec_xyz(instances[self.object2]["position"]))
        if d > 0.3:
            return 1, {"distance": d}
        else:
            return 0, {"distance": d}


# TODO: agent pass 和 agent at一样，可以合并
class PredicateAgentPass(Predicate):
    def __init__(self, object_id) -> None:
        self.object_id = object_id
        self.passed = False

    def task_done(self, action: Action, obs: Observation, options, task_setting) -> int:
        if self.passed:
            return 1, {}

        game_states = obs.game_states
        d = distance(vec_xz(game_states["agent"]["position"]), vec_xz(game_states["instances"][self.object_id]["position"]))
        if d < 0.3:
            self.passed = True
            return 1, {"distance": d}
        else:
            return 0, {"distance": d}


class PredicateSpecialActionSuccess(Predicate):
    def __init__(self, action_text) -> None:
        self.action_text = action_text
        self.passed = False

    def task_done(self, action: Action, obs: Observation, options, task_setting) -> int:
        if self.passed:
            return 1, {}
        if options[action.action_choice] == self.action_text:
            if obs.game_states["option_mode_info"]["feedback"] == "success":
                self.passed = True
                return 1, {}
        return 0, {}


def build_predicate(predicates, obs, old_version) -> List[Predicate]:
    pred_list = []
    print(predicates)
    for i in range(len(predicates)):

        def build_one_predicate(predicate):
            if old_version:
                if predicate.startswith("choose"):
                    return PredicateChoose(predicate.split(" ", maxsplit=1)[1])
                elif predicate.startswith("near"):
                    splits = predicate.split(" ")
                    assert len(splits) == 2
                    if len(splits) == 2:
                        return PredicateAgentNear(int(splits[1]))
                elif predicate.startswith("closer"):
                    splits = predicate.split(" ")
                    assert len(splits) == 3
                    return PredicateCloser(int(splits[1]), int(splits[2]), obs)
                elif predicate.startswith("further"):
                    splits = predicate.split(" ")
                    assert len(splits) == 3
                    return PredicateFurther(int(splits[1]), int(splits[2]), obs)
                elif predicate.startswith("on"):
                    splits = predicate.split(" ")
                    assert len(splits) == 3
                    return PredicateOn(int(splits[1]), int(splits[2]))
                elif predicate.startswith("grab_once"):
                    splits = predicate.split(" ")
                    assert len(splits) == 2
                    return PredicateGrabOnce(int(splits[1]))
                elif predicate.startswith("grab"):
                    splits = predicate.split(" ")
                    assert len(splits) == 2
                    return PredicateGrab(int(splits[1]))
                elif predicate.startswith("swap"):
                    splits = predicate.split(" ")
                    assert len(splits) == 3
                    return PredicateSwap(int(splits[1]), int(splits[2]), obs)
                elif predicate.startswith("in"):
                    splits = predicate.split(" ")
                    assert len(splits) == 3 or len(splits) == 4
                    return PredicateIn([int(_id) for _id in splits[1:-1]], int(splits[-1]))
                elif predicate.startswith("noton"):
                    splits = predicate.split(" ")
                    assert len(splits) == 3 or len(splits) == 4 or len(splits) == 5
                    return PredicateNotOn([int(_id) for _id in splits[1:-1]], int(splits[-1]))
            else:

                if predicate.startswith("choose"):
                    return PredicateChoose(predicate.split(" ", maxsplit=1)[1])

                elif predicate.startswith("agent_at"):
                    splits = predicate.split(" ")
                    if len(splits) == 3:
                        splits = splits[:-1]
                    assert len(splits) == 2
                    return PredicateAgentAt(int(splits[1]))
                elif predicate.startswith("agent_near"):
                    splits = predicate.split(" ")
                    if len(splits) == 3:
                        splits = splits[:-1]
                    assert len(splits) == 2
                    return PredicateAgentNear(int(splits[1]))
                elif predicate.startswith("agent_pass"):
                    splits = predicate.split(" ")
                    if len(splits) == 3:
                        splits = splits[:-1]
                    assert len(splits) == 2
                    return PredicateAgentPass(int(splits[1]))
                elif predicate.startswith("at"):
                    splits = predicate.split(" ")
                    assert len(splits) == 3
                    return PredicateAt(int(splits[1]), int(splits[2]))
                elif predicate.startswith("near"):
                    splits = predicate.split(" ")
                    assert len(splits) == 3
                    return PredicateNear(int(splits[1]), int(splits[2]))
                elif predicate.startswith("not_near"):
                    splits = predicate.split(" ")
                    assert len(splits) == 3
                    return PredicateNotNear(int(splits[1]), int(splits[2]))
                elif predicate.startswith("not_at"):
                    splits = predicate.split(" ")
                    assert len(splits) == 3
                    return PredicateNotAt(int(splits[1]), int(splits[2]))

                elif predicate.startswith("closer"):
                    splits = predicate.split(" ")
                    assert len(splits) == 3
                    return PredicateCloser(int(splits[1]), int(splits[2]), obs)
                elif predicate.startswith("further"):
                    splits = predicate.split(" ")
                    assert len(splits) == 3
                    return PredicateFurther(int(splits[1]), int(splits[2]), obs)
                elif predicate.startswith("grab_once"):
                    splits = predicate.split(" ")
                    if len(splits) == 3:
                        splits = splits[:-1]
                    assert len(splits) == 2
                    return PredicateGrabOnce(int(splits[1]))
                elif predicate.startswith("grab"):
                    splits = predicate.split(" ")
                    if len(splits) == 3:
                        splits = splits[:-1]
                    assert len(splits) == 2
                    return PredicateGrab(int(splits[1]))
                elif predicate.startswith("special_action_succes"):
                    splits = predicate.split(" ", maxsplit=1)
                    return PredicateSpecialActionSuccess(splits[1])

        pred_list.append(build_one_predicate(predicates[i]))
    return pred_list


def get_feedback(action: str, prev_obs, obs):
    return obs.game_states["option_mode_info"]["feedback"]
    action = action.lower()
    if action.startswith("grab"):
        if prev_obs.game_states["agent_grab_instance"] == -1 and obs.game_states["agent_grab_instance"] != -1:
            return "success"
        else:
            return "failed"
    elif action.startswith("put"):
        if prev_obs.game_states["agent_grab_instance"] != -1 and obs.game_states["agent_grab_instance"] == -1:
            return "success"
        else:
            return "failed"
    else:
        return "success"
