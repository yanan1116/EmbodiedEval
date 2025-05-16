from typing import List, Tuple, Union
from legent import Action
import re,os,sys
import queue
import threading
import numpy as np
import re
from prompts import PROMPT_IMAGE_PREFIX, PROMPT_VIDEO_PREFIX, PROMPT_SUFFIX
from openai import OpenAI,AzureOpenAI
import requests
import base64
import io
from PIL import Image
import time
import sys


VERBOSE = True


class AgentBase:
    def __init__(self, model_name: str, sync: bool, env) -> None:
        self.model_name = model_name
        self.images = []
        self.actions = []  # [(action, feedback)]
        self.task = ""
        self.sync = sync
        if not sync:
            self.env = env
            self.request_queue = queue.Queue()
            self.response_queue = queue.Queue()
            worker_thread = threading.Thread(target=self._process_request)
            worker_thread.daemon = True  # Set as a daemon thread so it will automatically quit if the main program exits
            worker_thread.start()
            self.is_waiting_response = False
        self.env = env

    def _process_request(self):
        while True:
            try:
                (image, feedback, options, video_path) = self.request_queue.get(timeout=1)  # Avoid endless polling of an empty queue, saving CPU resources, and ensures timely responses to new requests.

                action = self.act_sync(image, feedback, options, video_path)
                self.response_queue.put(action)
            except queue.Empty:
                continue

    def start(self, instruction, use_video):
        self.instruction = instruction
        self.use_video = use_video
        self.images = []
        self.actions = []

    def init_message(self):
        self.inputs = []

    def append_text(self, text):
        self.inputs.append(text)

    def append_image(self, image):
        self.inputs.append(image)

    def print_message(self):
        if VERBOSE:
            message = " ".join([t if type(t) == str else "<image>" for t in self.inputs])
            print("-" * 40 + "\n" + message + "-" * 40 + "\n")

    def generate(self):
        raise NotImplementedError

    def _act(self, actions, images, image, options, video_path=None):
        self.init_message()
        PROMPT_PREFIX = PROMPT_IMAGE_PREFIX if not self.use_video else PROMPT_VIDEO_PREFIX
        self.append_text(f"{PROMPT_PREFIX.format(self.instruction)}\n")
        self.append_text(f"Action history (action -> feedback):\n")
        for a in actions:
            self.append_text(f"\t{a[0]} -> {a[1]}\n")

        if not self.use_video:
            self.append_text(f"\nVisual history:\n")
            for o in images:
                self.append_image(o)

            self.append_text(f"\nCurrent view:\n")
            self.append_image(image)
        else:
            self.video_path = video_path

        options_string = "\n".join([f"{i}. {option}" for i, option in enumerate(
            options) if i > 0])  # remove the first idle option
        self.append_text(
            f"\n\n{PROMPT_SUFFIX.format(options_string, self.max_steps)}")

        self.print_message()

        return self.generate()

    def act(self, image, feedback, options, video_path):
        if self.sync:
            return self.act_sync(image, feedback, options, video_path)
        else:
            self.request_queue.put(
                (image, feedback, options, video_path))
            while True:
                self.env.step()
                try:
                    action = self.response_queue.get_nowait()
                    break
                except queue.Empty:
                    pass

            return action

    def act_sync(self, image, feedback, options, video_path):
        if feedback:
            self.update_feedback(feedback)

        result = self._act(self.actions, self.images, image,
                           options, video_path)
        if not result:
            print("failed to get response")
            return False
        try:
            payload, response = result["payload"], result["answer"]
            response = response.strip()
        except:
            response = result.strip()

        print("response:", response)
        action = Action()
        action.text = response

        try:
            # Match "Choice: [4]" or "Choice: 4" using a regular expression
            action.action_choice = int(
                re.search(r"Choice:?[\n\s]*\[?(\d+)\]?", response, re.IGNORECASE).group(1))
        except:
            try:
                action.action_choice = int(
                    re.search(r"(\d+)(?!.*\d)", response, re.DOTALL).group(1))
            except:
                action.action_choice = -1  # code for cannot match any number

        if action.action_choice > 0 and action.action_choice < len(options):
            self.update_history(image, options[action.action_choice])
        else:
            action.action_choice = -2  # code for option out of range

        return action

    def update_history(self, image, action):
        self.images.append(image)
        self.actions.append([action, ""])
        if len(self.images) > self.max_image_history:
            self.images = self.images[-self.max_image_history:]

    def update_feedback(self, feedback):
        if self.actions:
            self.actions[-1][1] = feedback


class AgentHuman(AgentBase):
    def __init__(self, env) -> None:
        super().__init__("human", env == None, env)
        self.env = env

    def act(self, image, feedback, options, video_path) -> int:
        action = Action()
        while True:
            obs = self.env.step()
            if obs.text != "":
                try:
                    action.action_choice = int(obs.text)
                    if action.action_choice < 0 or action.action_choice >= len(options):
                        continue
                except:
                    continue
                return action


class AgentRandom(AgentBase):
    def __init__(self, env) -> None:
        super().__init__("random", env == None, env)
        self.env = env

    def act(self, image, feedback, options, video_path) -> int:
        action = Action()
        action.action_choice = np.random.randint(1, len(options))
        return action


def encode_image(image):
    if type(image) == str:
        with open(image, "rb") as image:
            return base64.b64encode(image.read()).decode("utf-8")
    else:
        buffer = io.BytesIO()
        Image.fromarray(image).save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")


class AgentGPT(AgentBase):
    def __init__(self, env, model) -> None:
        super().__init__(model, env == None, env)
        self.model = model
        # self.api_key = os.environ['OPENAI_API_KEY']
        # self.base_url = "your base url"
        # self.client = OpenAI()
        self.client = AzureOpenAI(
            azure_endpoint = os.environ['AZURE_ENDPOINT'],  
            api_version= "2024-10-01-preview",
            api_key = os.environ['AZURE_OPENAI_API_KEY'] if self.model in ['gpt-4o-mini'] else os.environ['AZURE_OPENAI_API_KEY_41']
            )
    def generate(self):
        # Organize the inputs (text and image list) into a payload for ChatGPT.
        messages = [{"role": "user", "content": []}]
        for input in self.inputs:
            if type(input) == str:
                messages[0]["content"].append({"type": "text", "text": input})
            elif type(input) == np.ndarray:
                messages[0]["content"].append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image(input)}"}})

        # def send_request(messages):
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=1024,
            temperature=0
        )
        answer = completion.choices[0].message.content.strip()
        return answer


# TODO: Overwrite this class for evaluating your own model.
class MyAgent(AgentBase):
    def __init__(self, env) -> None:
        super().__init__("your_model_name", env == None, env)
        # Place the initialization code for your model/API here
        # self.model = ...

    def generate(self):
        return "Choice: [1]"
        # ===================== For Image Model =====================
        inputs: List[Union[str, np.ndarray]] = self.inputs
        # Prepare the input for your model. For example:
        #   texts = [t for t in inputs if type(t) == str else "<image>"]
        #   images = [i for i in inputs if type(i) == np.ndarray]
        #   text = "".join(texts)

        # Perform model inference and return the output text. For example:
        #   return self.model.generate(images, text)

        # ===================== For Video Model =====================
        video_path: str = self.video_path
        video_frames: List[np.ndarray] = self.frames
        text: str = "".join([t for t in self.inputs if type(t) == str])
        # Prepare the input for your model. For example:
        #   video = self.processor(video_frames)

        # Perform model inference and return the output text.
        #   return self.model.generate(video, text)
