# How to use:
# python run_eval.py --agent random --max_steps 24 --max_images 24 --port 50051 --all
# python run_eval.py --agent human --max_steps 24 --max_images 24 --port 50051 --all

import os
import re
import argparse
from legent import (Environment, ActionFinish, store_json, load_json, 
                   ResetInfo, save_image, time_string)
from legent.utils.math import distance, vec_xz
from legent.utils.io import log_green, create_video
from legent.action.api import SetVideoRecordingPath
from predicate import build_predicate, get_feedback
from agent import *
from task_setup import process_task_settings
from sys import platform

MAX_STAY_COUNT = 100

def get_platform():
    if platform == "linux" or platform == "linux2":
        platform_name = "Linux"
    elif platform == "darwin":
        platform_name = "MacOS"
    elif platform == "win32":
        platform_name = "Windows"
    else:
        print("Cannot decide platform. Exit.")
        exit(0)
    return platform_name

def initialize_environment(port, use_video, remote):
    root_folder = "data/envs"
    envs_path = [os.path.join(root_folder, d) for d in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, d))]
    path = envs_path[0]
    if remote:
        path = None
    return Environment(
        env_path=path, action_mode=1, camera_resolution_width=448,
        camera_resolution_height=448, camera_field_of_view=90,
        run_options={"port": port, "width": 768, "height": 768},
        use_animation=use_video, rendering_options={"use_default_light": 1, "style": 0}
    )

def create_agent(agent_type, sync, env):
    agents = {
        "human": lambda: AgentHuman(env),
        "random": lambda: AgentRandom(env),
        "gpt-4o": lambda: AgentGPT4o(None if sync else env),
        "myagent": lambda: MyAgent(None if sync else env)
    }
    if agent_type not in agents: 
        raise ValueError(f"Unsupported agent type: {agent_type}")
    return agents[agent_type]()

def load_task_data(scene_folder, run_one_task_instance):
    tasks = load_json("data/tasks/tasks.json")
    task_to_type = {i: t["task_file"].split("/")[0] for i, t in enumerate(tasks)}
    all_paths = [run_one_task_instance] if run_one_task_instance else ["data/tasks/"+t["task_file"] for t in tasks]
    return task_to_type, process_task_settings(all_paths, scene_folder, tasks)

def initialize_episode(task_i, task_setting, agent, env, save_path, use_video):
    print("\n" + "==" * 8 + f"Start episode {task_i}" + "==" * 8)
    agent.start(task_setting["task"], use_video)
    print(task_setting["task"])
    obs = env.reset(ResetInfo(scene=task_setting["scene"], api_calls=[]))
    traj_save_dir = f"{save_path}/traj{task_i:04d}"
    os.makedirs(traj_save_dir)
    store_json(task_setting["task_raw"], f"{traj_save_dir}/task_raw.json")
    store_json(task_setting, f"{traj_save_dir}/task.json")
    save_image(obs.image, f"{traj_save_dir}/{0:04d}.png")
    if use_video:
        create_video([obs.image], f"{traj_save_dir}/{0:04d}.mp4", fps=1)
    return obs, 0, 0, [], 0, obs.game_states["agent"]["position"], traj_save_dir

def process_predicates(task_setting, obs, run_one_task_instance, run_all_task_instance):
    if run_one_task_instance or run_all_task_instance:
        task_setting["predicates"] = [re.sub(r"\s+", " ", p) 
                                      for p in obs.game_states["option_mode_info"]["predicates"]]
        print(task_setting["scene"]["task_instance"]["task_text"])
    print("Predicates:", task_setting["predicates"])
    return build_predicate(task_setting["predicates"], obs, not run_one_task_instance and not run_all_task_instance)

def execute_action(agent, obs, feedback, options, use_video, traj_save_dir, step):
    action = agent.act(obs.image, feedback, options, f"{traj_save_dir}/{step:04d}.mp4")
    error = ""
    response = action.text if action else ""
    thought = ""
    if action:
        try:
            thought = re.search(r"Thought: *(.*?)\nChoice:", response, re.DOTALL).group(1).strip()
        except:
            pass
        if action.action_choice < 0:
            error = "no option match" if action.action_choice == -1 else "option out of range"
            log_green(error)
        action.text = ""
        if use_video:
            action.api_calls = [SetVideoRecordingPath(f"{traj_save_dir}/frames_client/{step + 1:04d}_")]
    else:
        error = "api_crash" 
    return action, error, response, thought

def step_environment(env, action, use_video, traj_save_dir, step, frames):
    obs = env.step(action)
    if use_video:
        frames_folder = f"{traj_save_dir}/frames"
        os.makedirs(frames_folder, exist_ok=True)
        for i, frame in enumerate(obs.frames):
            save_image(frame, f"{frames_folder}/{step + 1:04d}_{i:04d}.png")
        log_green(f"frames {len(obs.frames)}")
        frames.extend(obs.frames)
        create_video(frames, f"{traj_save_dir}/{step + 1:04d}.mp4", 30)
    return obs, frames

def evaluate_tasks(agent, max_steps, max_images, port, scene_folder, save_path, 
                   task_ids, sync, run_one_task_instance, run_all_task_instance, use_video, remote):
    
    MAX_IMAGE_HISTORY = max_images - 1
    failed_cases, success_cases = [], []
    task_to_type, task_settings = load_task_data(scene_folder, run_one_task_instance)
    if not task_ids:
        task_ids = list(range(len(task_settings)))
   
    
    env = initialize_environment(port, use_video, remote)
    
    save_path = save_path or f"results/{time_string()}-{agent}-case{task_ids[0]}"
    os.makedirs(save_path)
    store_json(task_ids, f"{save_path}/task_ids.json")
    store_json({"agent": agent, "max_steps": max_steps, "max_images": max_images}, f"{save_path}/run_args.json")
    
    try:
        agent = create_agent(agent, sync, env)
        agent.max_steps = max_steps
        agent.max_image_history = MAX_IMAGE_HISTORY
        success_count = 0
        
        for task_i in task_ids:
            task_setting = task_settings[task_i]
            obs, step, done, frames, stuck_count, stuck_pos, traj_save_dir = initialize_episode(task_i, task_setting, agent, env, save_path, use_video)
            task_category = task_to_type[task_i]
            pred_list = process_predicates(task_setting, obs, run_one_task_instance, run_all_task_instance)
            options = obs.game_states["option_mode_info"]["options"]
            feedback, prev_obs = None, obs
            print(options)
            
            while step < max_steps:
                if use_video:
                    agent.frames = frames
                action, error, response, thought = execute_action(agent, obs, feedback, options, use_video, traj_save_dir, step)
                if error:
                    store_json({"step": step, "options": options, "action_choice": action.action_choice, "action_error": error, "action": None, "response": response, "thought": thought, "done_after_action": done, "info_after_action": info, "feedback": feedback, "predicates_done": done_list, "time": time_string()}, f"{traj_save_dir}/{step:04d}a.json")
                    break

                obs, frames = step_environment(env, action, use_video, traj_save_dir, step, frames)
                new_options = obs.game_states["option_mode_info"]["options"]
                feedback = get_feedback(options[action.action_choice], prev_obs, obs)
                feedback_content = obs.game_states["option_mode_info"]["feedback_content"]
                prev_obs = obs
                save_image(obs.image, f"{traj_save_dir}/{step + 1:04d}.png")
                print(f"step {step}, action: {action.action_choice}. {options[action.action_choice]}, feedback: {feedback} - {feedback_content}\n")
                feedback = feedback + (f": {feedback_content}" if feedback_content else "")

                done = 1
                done_list = []
                for predicate in pred_list:
                    _done, info = predicate.task_done(action, obs, options, task_setting)
                    done_list.append(_done)
                    if _done == -1:
                        done = -1
                        break
                    elif _done == 0:
                        done = 0
                print(f"goal complete ratio: {done_list.count(1)} / {len(done_list)}")

                if distance(vec_xz(stuck_pos), vec_xz(obs.game_states["agent"]["position"])) < 0.01:
                    stuck_count += 1
                else:
                    stuck_count = 0
                stuck_pos = obs.game_states["agent"]["position"]
                if stuck_count > MAX_STAY_COUNT:
                    done = -1

                store_json({"step": step, "options": options, "action_choice": action.action_choice, "action": options[action.action_choice], "response": response, "thought": thought, "done_after_action": done, "info_after_action": info, "feedback": feedback, "predicates_done": done_list, "time": time_string()}, f"{traj_save_dir}/{step:04d}a.json")
                options = new_options
                print(options)
                step += 1

                if step == max_steps - 1 and "QA" in task_category:
                    options = [option for option in options if "answer" in option]
                if done == 1:
                    success_count += 1
                    log_green("Task accomplished.")
                if isinstance(action, ActionFinish) or action.text or done != 0:
                    save_image(obs.image, f"{traj_save_dir}/{step:04d}.png")
                    break

            if done != 1:
                failed_cases.append(task_i)
                store_json({"result": "failed", "steps_taken": step}, f"{traj_save_dir}/result.json")
                log_green("Task failed.")
            else:
                success_cases.append(task_i)
                store_json({"result": "success", "steps_taken": step}, f"{traj_save_dir}/result.json")

            log_green(f"success rate: {success_count}/{len(success_cases) + len(failed_cases)} of {len(task_settings)}")
            result = {
                "Success Rate": f"{success_count}/{len(success_cases) + len(failed_cases)}",
                "test cases": task_ids,
                "failed cases": failed_cases,
                "success cases": success_cases,
            }
            if not run_one_task_instance:
                print(result)
            store_json(result, f"{save_path}/result_temp.json")
            if run_one_task_instance:
                break

    except Exception as e:
        print("Exception:", e)
        raise e
    finally:
        env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", type=str, default="gpt-4o")
    parser.add_argument("--test_case_start", type=int, default=-1)
    parser.add_argument("--test_case_end", type=int, default=328)
    parser.add_argument("--max_steps", type=int, default=24)
    parser.add_argument("--max_images", type=int, default=25)
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--scene_folder", type=str, default="data/scenes") # TODO make it a fixed value
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--sync", action="store_true")
    parser.add_argument("--run_one_task_instance", type=str, default=None)
    parser.add_argument("--all", type=bool, default=True)
    parser.add_argument("--use_video", action="store_true")
    parser.add_argument("--remote", action="store_true")
    args = parser.parse_args()
    task_ids = list(range(args.test_case_start, args.test_case_end)) if args.test_case_start != -1 and args.test_case_end != -1 else None
    evaluate_tasks(args.agent, args.max_steps, args.max_images, args.port, args.scene_folder, args.save_path, task_ids, args.sync, args.run_one_task_instance, args.all, args.use_video, args.remote)