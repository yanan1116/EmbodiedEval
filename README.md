# EmbodiedEval: Evaluate Multimodal LLMs as Embodied Agents
<p align="center">
   <a href="https://embodiedeval.github.io" target="_blank">🌐 Project Page</a> | <a href="https://huggingface.co/datasets/EmbodiedEval/EmbodiedEval" target="_blank">🤗 Dataset</a> | <a href="" target="_blank">📃 Paper </a>
</p>

**EmbodiedEval** is a comprehensive and interactive benchmark designed to evaluate the capabilities of MLLMs in embodied tasks.

## Installation

### Setup Simulation Environment

EmbodiedEval includes a 3D simulator for realtime simulation. You have two options to run the simulator:

Option 1: Run the simulator on your personal computer with a display (Windows/MacOS/Linux). No additional configuration is required. The subsequent installation and data download (approximately 20GB of space) will take place on your computer.

Option 2: Run the simulator on a Linux server, which requires sudo access, up-to-date NVIDIA drivers, and running outside a Docker container. Additional configurations are required as follows:

<details>
  <summary>Additional configurations</summary>
<br>

1. Install Xorg:

    ```
    sudo apt install -y gcc make pkg-config xorg
    ```

2. Generate .conf file:

    ```
    sudo nvidia-xconfig --no-xinerama --probe-all-gpus  --use-display-device=none
    sudo cp /etc/X11/xorg.conf /etc/X11/xorg-0.conf
    ```

3. Edit /etc/X11/xorg-0.conf:

    - Remove "ServerLayout" and "Screen" section.
    - Set `BoardName` and `BusID` of "Device" section to    the corresponding `Name` and `PCI BusID` of a GPU  displayed by the `nvidia-xconfig --query-gpu-info`   command. For example:
        ```
        Section "Device"
            Identifier     "Device0"
            Driver         "nvidia"
            VendorName     "NVIDIA Corporation"
            BusID          "PCI:164:0:0"
            BoardName      "NVIDIA GeForce RTX 3090"
        EndSection
        ```

4. Run Xorg:

    ```
    sudo nohup Xorg :0 -config /etc/X11/xorg-0.conf &
    ```

5. Set the display (Remember to run the following command in every new terminal session before running the evaluation code):

    ```
    export DISPLAY=:0
    ```
</details>


### Install Dependencies

```bash
conda create -n embodiedeval python=3.10
conda activate embodiedeval
pip install -r requirements.txt
```

### Download Dataset

```bash
python download.py
```


## Evaluation

### Run Baselines

#### Random baseline

```bash
python run_eval.py --agent random
```

#### Human baseline

```bash
python run_eval.py --agent human
```

In human baseline, you can manually interact with the environment.
<details>
 <summary>How to play</summary>
<br>

- Use the keyboard to press the corresponding number to choose an option;

- Pressing W/A/D will map to the forward/turn left/turn right options in the menu;

- Pressing Enter opens or closes the chat window, and you can enter option numbers greater than 9;

- Pressing T will hide/show the options panel.
</details>

#### GPT-4o

Edit the `api_key` and `base_url` in agent.py and run:
```bash
python run_eval.py --agent gpt-4o
```

### Evaluate Your Own Model

To evaluate your own model, you need to overwrite the `MyAgent` class in `agent.py`. 
In the `__init__` method, you need to load the model or initialize the API. 
In the `generate` method, you need to perform model inference or API calls and return the generated text. See the comments within the class for details.

Run the following code to evaluate your model.
```bash
python run_eval.py --agent myagent
```

If your server cannot run the simulator (e.g. without sudo access), and your personal computer cannot run the model. You can run simulation on your computer and the model on the server using the following steps:
<details>
<summary>Evaluation steps with a remote simulator</summary>
<br>

1. Perform the `Install Dependencies` and `Download Dataset` steps on both your local computer and the server.

2. On the server, run:
    ```
    python run_eval.py --agent myagent  --remote --scene_folder <The     absolute path of the scene folder   on your local computer>
    ```
    This command will hang, waiting     for the simulator to connect.


3. On your computer, set up a SSH tunnel between your computer and the server: 
    ```
    ssh -N -L 50051:localhost:50051     <username>@<host> [-p <ssh_port>]
    ```

4. On your computer, launch the simulator:
    ```
    python launch.py
    ```

    Once the simulator starts, the  evaluation process on the server     will begin.

</details>


### Compute Metrics

Run metrics.py with the result folder as a parameter to compute the performance. The `total_metrics.json` (overall performance) and `type_metrics.json` (performance per task type) will be saved in the result folder.

```
python metrics.py --result_folder results/xxx-xxx-xxx
```

### Citation

```
```