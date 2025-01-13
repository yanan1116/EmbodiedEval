from legent.environment.env_utils import validate_environment_path, launch_executable
import subprocess
import os
root_folder = "data/envs"
envs_path = [os.path.join(root_folder, d) for d in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, d))]
env_path = envs_path[0]

run_args = ["--width", "768", "--height",  "768", "--port", "50051"]


def launch_executable(file_name: str, args) -> subprocess.Popen:
    launch_string = validate_environment_path(env_path)
    launch_string = launch_string.replace("\\", "/")

    subprocess_args = [launch_string] + args
    print(subprocess_args)
    # std_out_option = DEVNULL means the outputs will not be displayed on terminal.
    # std_out_option = None is default behavior: the outputs are displayed on terminal.
    std_out_option = subprocess.DEVNULL
    try:
        return subprocess.Popen(
            subprocess_args,
            start_new_session=True,
            stdout=std_out_option,
            stderr=std_out_option,
        )
    except PermissionError as perm:
        # This is likely due to missing read or execute permissions on file.
        raise Exception("EnvironmentException:\n" f"Error when trying to launch environment - make sure " f"permissions are set correctly. For example " f'"chmod -R 755 {launch_string}"') from perm


launch_executable(file_name=env_path, args=run_args)
