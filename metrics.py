import argparse
import os
from legent import load_json, store_json
import re

index2json_path = "data/tasks/tasks.json"
index2json = load_json(index2json_path)
index2json = {str(i["index"]): i["task_file"] for i in index2json}

def calculate_goal_conditioned_success(goal_condition):
    """
    Calculate the goal-conditioned success rate based on the given list of final goal conditions.
    """
    total_goals = len(goal_condition)
    return goal_condition.count(1) / total_goals

def list_folders(path):
    folders = []
    for entry in os.listdir(path):
        entry_path = os.path.join(path, entry)
        if os.path.isdir(entry_path):
            folders.append(entry)
    return folders

def aggregate_json_files(folder_path):
    aggregated_data = []
    for filename in os.listdir(folder_path):
        if filename.endswith("a.json"):
            file_path = os.path.join(folder_path, filename)
            data = load_json(file_path)
            aggregated_data.append(data)
    return aggregated_data

def traj_to_index(traj_name):
    match = re.match(r"traj(\d+)", traj_name)
    if match:
        return int(match.group(1))
    else:
        raise ValueError(f"Invalid trajectory name: {traj_name}")

def safe_divide(numerator, denominator):
    return round((numerator / denominator) * 100, 2) if denominator > 0 else 0

def compute_metrics_for_each_type(results_folder, task_type_folder_list, human_traj_folder, max_step=24):
    max_step_exceeded, success, failure, errors, traj_lengths, no_option_match, option_out_of_range, api_crash = [], [], [], [], [], [], [], []
    interaction_success_count, interaction_total_count = 0, 0 
    spls = [] 
    total_goal_condition_success = 0
    compute_spl = True

    for episode_folder in task_type_folder_list:
        if os.path.exists(f"{results_folder}/{episode_folder}/traj.json"):
            traj_list = load_json(f"{results_folder}/{episode_folder}/traj.json")
        else:
            traj_list = aggregate_json_files(f"{results_folder}/{episode_folder}")
            store_json(traj_list, f"{results_folder}/{episode_folder}/traj.json")


        if os.path.exists(f"{human_traj_folder}/{episode_folder}/traj.json"):
            optimal_traj = load_json(f"{human_traj_folder}/{episode_folder}/traj.json")
        else:
            compute_spl = False
            
        traj_lengths.append(len(traj_list))
        done_status = traj_list[-1]["done_after_action"]
        
        final_traj = traj_list[-1]
        if done_status == 0 and len(traj_list) >= max_step and final_traj["action_choice"] not in [-1, -2, None]: # 超出一定的步长
            max_step_exceeded.append(episode_folder)
        if final_traj["action_choice"] == -1 or ("action_error" in final_traj and final_traj["action_error"] == "no option match"):
            no_option_match.append(episode_folder)
        elif final_traj["action_choice"] == -2 or ("action_error" in final_traj and final_traj["action_error"] == "option out of range"):
            option_out_of_range.append(episode_folder)
        elif final_traj["action_choice"] == None or ("action_error" in final_traj and final_traj["action_error"] == "api_crash"):
            api_crash.append(episode_folder)

        if done_status == 1:
            success.append(episode_folder)
        elif done_status == -1:
            failure.append(episode_folder)
        else:
            errors.append(episode_folder)
            
        # Compute SPL
        if compute_spl:
            S_i = 1 if done_status == 1 else 0
            L_i = len(optimal_traj)
            P_i = len(traj_list)
            denominator = max(L_i, P_i)
            spl_i = S_i * (L_i / denominator) if denominator > 0 else 0
            spls.append(spl_i)
        else:
            spls = [0] * len(traj_lengths)
        
        # calculate_goal_conditioned_success
        predicates = load_json(f"{results_folder}/{episode_folder}/task.json")["scene"]["task_instance"]["predicates"]
        if len(predicates) == 1:
            final_predicates_done = [done_status]
        else:
            final_predicates_done = traj_list[-1]["predicates_done"]

        goal_conditioned_success = calculate_goal_conditioned_success(final_predicates_done)
        total_goal_condition_success += goal_conditioned_success

        interaction_traj = [step for step in traj_list if step.get("feedback")]
        interaction_total_count += len(interaction_traj)
        interaction_success_count += sum(1 for step in interaction_traj if step["feedback"] != "failed") 

    total_episodes = len(success) + len(failure) + len(errors)
    # assert len(errors) == len(max_step_exceeded) + len(no_option_match) + len(option_out_of_range) + len(api_crash)

    accuracy = safe_divide(len(success), total_episodes)
    max_exceed_rate = safe_divide(len(max_step_exceeded), total_episodes)
    fail_rate = safe_divide(len(failure), total_episodes)  
    no_option_match_rate = safe_divide(len(no_option_match), total_episodes)
    option_out_of_range_rate = safe_divide(len(option_out_of_range), total_episodes)
    api_crash_rate = safe_divide(len(api_crash), total_episodes) 
    avg_traj_length = safe_divide(sum(traj_lengths), total_episodes) 
    avg_goal_condition_success = safe_divide(total_goal_condition_success, total_episodes)
    interaction_accuracy = safe_divide(interaction_success_count, interaction_total_count)
    average_spl = safe_divide(sum(spls), total_episodes)

    return {
        "accuracy": accuracy,
        "max_step_exceed_rate": max_exceed_rate,
        "fail_rate": fail_rate,  
        "no_option_match_rate": no_option_match_rate, 
        "option_out_of_range_rate": option_out_of_range_rate,
        "api_crash_rate": api_crash_rate, 
        "average_trajectory_length": round(avg_traj_length, 2),
        "average_spl": average_spl, 
        "total_episodes": total_episodes,
        "success_count": len(success),
        "failure_count": len(failure),
        "max_step_exceeded_count": len(max_step_exceeded),
        "interaction_accuracy": interaction_accuracy,
        "total_interactions": interaction_total_count,
        "interaction_rate": safe_divide(interaction_total_count, total_episodes),
        "successful_interactions": interaction_success_count,
        "avg_goal_condition_success": avg_goal_condition_success,
        "success": success,
        "failure": failure,
        "max_exceed": max_step_exceeded
    }

def compute_metrics_for_all_types(total_result_folder, model_name, human_traj_folder, max_step=24):
    total_metrics = {
        "model_name": model_name,
        "accuracy": 0.0,
        "max_step_exceed_rate": 0.0,
        "fail_rate": 0.0, 
        "no_option_match_rate": 0.0, 
        "option_out_of_range_rate": 0.0, 
        "api_crash_rate": 0.0, 
        "average_trajectory_length": 0.0,
        "average_spl": 0.0,
        "total_episodes": 0,
        "success_count": 0,
        "failure_count": 0,
        "max_step_exceeded_count": 0,
        "interaction_accuracy": 0.0,
        "total_interactions": 0,
        "successful_interactions": 0,
        "avg_goal_condition_success": 0.0
    }

    type_metrics = []
    task_type_folder_dict = {}

    for episode_folder in list_folders(total_result_folder):
        index = str(traj_to_index(episode_folder))
        json_path = index2json[index]
        task_type = json_path.split("/")[0]
        
        if task_type in task_type_folder_dict:
            task_type_folder_dict[task_type].append(episode_folder)
        else:
            task_type_folder_dict[task_type] = [episode_folder]

    for task_type in task_type_folder_dict:
        task_type_folder_list = task_type_folder_dict[task_type]
        metrics = compute_metrics_for_each_type(total_result_folder, task_type_folder_list, human_traj_folder, max_step)
        type_metrics.append({
            "model_name": model_name, 
            "task_type": task_type,
            **{k: v for k, v in metrics.items() if k not in ["success", "failure", "max_exceed"]},  
            "success": metrics["success"], 
            "failure": metrics["failure"],
            "max_exceed": metrics["max_exceed"]
        })

        total_metrics["accuracy"] += metrics["accuracy"] * metrics["total_episodes"]
        total_metrics["avg_goal_condition_success"] += metrics["avg_goal_condition_success"] * metrics["total_episodes"]
        total_metrics["max_step_exceed_rate"] += metrics["max_step_exceed_rate"] * metrics["total_episodes"]
        total_metrics["fail_rate"] += metrics["fail_rate"] * metrics["total_episodes"] 
        total_metrics["no_option_match_rate"] += metrics["no_option_match_rate"] * metrics["total_episodes"] 
        total_metrics["option_out_of_range_rate"] += metrics["option_out_of_range_rate"] * metrics["total_episodes"] 
        total_metrics["api_crash_rate"] += metrics["api_crash_rate"] * metrics["total_episodes"] 
        total_metrics["average_trajectory_length"] += metrics["average_trajectory_length"] * metrics["total_episodes"]
        total_metrics["average_spl"] += metrics["average_spl"] * metrics["total_episodes"]
        total_metrics["total_episodes"] += metrics["total_episodes"]
        total_metrics["success_count"] += metrics["success_count"]
        total_metrics["failure_count"] += metrics["failure_count"]
        total_metrics["max_step_exceeded_count"] += metrics["max_step_exceeded_count"]
        total_metrics["interaction_accuracy"] += metrics["interaction_accuracy"] * metrics["total_episodes"]
        total_metrics["total_interactions"] += metrics["total_interactions"]
        total_metrics["successful_interactions"] += metrics["successful_interactions"]

    if total_metrics["total_episodes"] > 0:
        for key in ["accuracy", "avg_goal_condition_success", "max_step_exceed_rate", "fail_rate", "no_option_match_rate", "option_out_of_range_rate", "api_crash_rate", "average_trajectory_length", "interaction_accuracy", "average_spl"]:
            total_metrics[key] /= total_metrics["total_episodes"]

    total_metrics["interaction_rate"] = safe_divide(total_metrics["total_interactions"], total_metrics["total_episodes"])
    return total_metrics, type_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_folder", type=str)
    parser.add_argument("--model_name", type=str, default="")
    args = parser.parse_args()
    
    model_name = args.model_name
    total_result_folder = args.result_folder
    human_traj_folder = ""
    total_metrics, type_metrics = compute_metrics_for_all_types(total_result_folder, model_name, human_traj_folder)
    # store the result to results_folder:
    store_json(total_metrics, f"{total_result_folder}/total_metrics.json")
    store_json(type_metrics, f"{total_result_folder}/type_metrics.json")