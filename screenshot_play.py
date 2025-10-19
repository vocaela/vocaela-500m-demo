# Given a screenshot image play the instructions
import json
from logging import Logger
from pathlib import Path
from time import strftime
from typing import Any, Dict, List
from PIL import ImageFont
import torch
import fire

from transformers import AutoModelForImageTextToText, AutoProcessor

Vocaelam_Computer_Use_System_Message = """
You are an assistant trained to navigate the computer screen. 
Given a task instruction, a screen observation, and an action history sequence, 
output the next actions and wait for the next observation. 

## Allowed ACTION_TYPEs and parameters:
1. `PRESS_KEY`: Press one specified key. Two parameters: `key`, string, the single key to press; `presses`, integer, the number of times to press the key (default is 1).
2. `TYPE`: Type a string into an element. Parameter: `text`, string, the text to type.
3. `MOUSE_MOVE`: Move the mouse cursor to a specified position. Parameter: `coordinate`, formatted as [x,y], the position to move the cursor to.
4. `CLICK`: Click left mouse button once on an element. Parameter: `coordinate`, formatted as [x,y], the position to click on.
5. `DRAG`: Drag the cursor with the left mouse button pressed, start and end positions are specified. Two parameters: `coordinate`, formatted as [x,y], the start position to drag from; `coordinate2`, formatted as [x2,y2], the end position to drag to.
6. `RIGHT_CLICK`: Click right mouse button once on an element. Parameter: `coordinate`, formatted as [x,y], the position to right click on.
7. `MIDDLE_CLICK`: Click middle mouse button once on an element. Parameter: `coordinate`, formatted as [x,y], the position to middle click on.
8. `DOUBLE_CLICK`: Click left mouse button twice on an element. Parameter: `coordinate`, formatted as [x,y], the position to double click on.
9. `SCROLL`: Scroll the screen (via mouse wheel). Parameter: `scroll_direction`, the direction (`up`/`down`/`left`/`right`) to scroll.
10. `HOTKEY`: Press a combination of keys simultaneously. Parameter: `hotkeys`, list of strings, the keys to press together.

* NOTE *: The `coordinate` and `coordinate2` parameters (formatted as [x,y]) are the relative coordinates on the screenshot scaled to range of 0-1, [0,0] is the top-left corner and [1,1] is the bottom-right corner.

## Format your response as
<Action>the next actions</Action>

`The next actions` can be one or multiple actions. Format `the next actions` as a JSON array of objects as below, each object is an action:
[{"action": "<ACTION_TYPE>", "key": "<key>", "presses": <presses>, "hotkeys": ["<hotkeys>"], "text": "<text>", "coordinate": [x,y], "coordinate2": [x2,y2],  "scroll_direction": "<scroll_direction>"}]

If a parameter is not applicable, don't include it in the JSON object.
"""

Vocaela_Mobile_Use_System_Message = """
You are an assistant trained to navigate the mobile phone. 
Given a task instruction, a screen observation, and an action history sequence, 
output the next actions and wait for the next observation. 

## Allowed ACTION_TYPEs and parameters:
1. `CLICK`: Click/tap on the screen. Parameter: `coordinate`, formatted as [x,y], the position to click on.
2. `LONG_PRESS`: Long press on the screen. Two parameters: `coordinate`, formatted as [x,y], the position to long press on; `time`, duration in seconds to long press.
3. `SWIPE`: Swipe on the screen. Two parameters: `swipe_from`, the start area to swipe from, only allowed value in {'top', 'bottom', 'left', 'right', 'center', `top_left`, `top_right`, `bottom_left`, `bottom_right`}; `swipe_direction`, the direction (`up`/`down`/`left`/`right`) to swipe towards.
4. `TYPE`: Type a string into an element. Parameter: `text`, string, the text to type.
5. `SYSTEM_BUTTON`: Press a system button. Parameter: `button`, the system button to press, allowed button values: 'Back', 'Home', 'Menu', 'Enter'.
6. `OPEN`: Open an app. Parameter: `text`, string, the app name to open.

* NOTE *: The `coordinate` parameter (formatted as [x,y]) is the relative coordinates on the screenshot scaled to range of 0-1, [0,0] is the top-left corner and [1,1] is the bottom-right corner.

## Format your response as
<Action>the next actions</Action>

`The next actions` can be one or multiple actions. Format `the next actions` as a JSON array of objects as below, each object is an action:
[{"action": "<ACTION_TYPE>", "text": "<text>", "coordinate": [x,y], "swipe_from": "<swipe_from>", "swipe_direction": "<swipe_direction>", "button": "<button>"}]

If a parameter is not applicable, don't include it in the JSON object.
"""

_DEFAULT_MAX_NEW_TOKENS: int = 512
_DEFAULT_TEMPERATURE: float = 0.0
_DEFAULT_TOP_P: float = 1.0
_DEFAULT_TOP_K: int = -1
_DEFAULT_DTYPE: str = "float16"

class HFInferenceClient:
    def __init__(
            self, 
            model_path: str, 
            temperature: float = _DEFAULT_TEMPERATURE,
            max_new_tokens: int = _DEFAULT_MAX_NEW_TOKENS,
            top_p: float = _DEFAULT_TOP_P,
            top_k: int = _DEFAULT_TOP_K,
            stop_strings: List[str]|str = None,
            output_skip_special_tokens: bool = False, # output all tokens for transparency, set to True to skip them.
            torch_dtype: torch.dtype = torch.float16,
            device: str = "cuda" if torch.cuda.is_available() else "cpu",
            logger: Logger = None,
            ):
        self.model_path = model_path
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.top_p = top_p
        self.top_k = top_k
        self.stop_strings = stop_strings
        self.output_skip_special_tokens = output_skip_special_tokens

        generate_kwargs = {
            "temperature": temperature,
            "do_sample": temperature > 0.0,
            "max_new_tokens": max_new_tokens,
            "top_p": top_p,
            "top_k": top_k,
            "stop_strings": stop_strings
        }

        self.generate_kwargs = {k: v for k, v in generate_kwargs.items() if v is not None}
        if logger:
            logger.info(f"Generation kwargs: {self.generate_kwargs}")
        
        self.torch_dtype = torch_dtype
        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = AutoModelForImageTextToText.from_pretrained(model_path, torch_dtype=self.torch_dtype).to(self.device)

    def predict(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device, dtype=self.torch_dtype)

        generated_ids = self.model.generate(
            **inputs, 
            **self.generate_kwargs
        )
        
        # remove prompt from gen. tokens
        outputs_tokenized=[tok_out[len(tok_in):] 
            for tok_in, tok_out in zip(inputs["input_ids"], generated_ids)]

        generated_texts = self.processor.batch_decode(
            outputs_tokenized,
            skip_special_tokens=self.output_skip_special_tokens,
        )

        return {
            "text": generated_texts[0],
        }


def create_messages(image_path, instruction, system_message: str):
    messages = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": system_message}
            ]
        },
        {
            "role": "user", 
            "content": [
                {
                    "type": "image",
                    "url": image_path
                },
                {"type": "text", "text": instruction}
            ]
        },
    ]

    return messages

def parse_actions_from_output(output: str):
    start = output.find("<Action>")
    end = output.find("</Action>")
    if start == -1 or end == -1 or start >= end:
        raise ValueError("Output does not contain valid <Action>...</Action> tags.")
    
    actions_str = output[start + len("<Action>"):end].strip()
    return json.loads(actions_str)

def draw_groundings_on_image(image_path, output_text: str, output_path):
    from PIL import Image, ImageDraw
    import ast

    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size
    # first draw actions text on top-left corner
    font = ImageFont.truetype("DejaVuSans.ttf", size=12)  # ← control size here
    text_x, text_y = 10, 10
    bbox = draw.textbbox((text_x, text_y), output_text, font=font)
    # Draw rectangle background
    draw.rectangle(bbox, fill="black")
    draw.text((text_x, text_y), output_text, fill="white", font=font)
    try:
        action_objs = parse_actions_from_output(output_text)
    except Exception as e:
        print(f"Failed to parse actions from output: {e}")
        image.save(output_path)
        return
    
    def draw_coordinate(coord, color, r=15):
        if isinstance(coord, list) and len(coord) == 2:
            x = int(coord[0] * width)
            y = int(coord[1] * height)
            draw.ellipse((x - r, y - r, x + r, y + r), outline=color, width=8)
            return x, y
        return None
    
    for action in action_objs:
        # get coordinate and coordinate2 if applicable, draw a circle for each coordinate
        if "coordinate" in action:
            coord = action["coordinate"]
            draw_coordinate(coord, "red")
        if "coordinate2" in action:
            coord2 = action["coordinate2"]
            draw_coordinate(coord2, "blue")

    image.save(output_path)

_help_msg = """Enter instruction or command:
  - <instruction>: directly type your instruction to interact with the screenshot 
  - /img <image **full path**>: specify screenshot image **full path**
  - /desktop: switch to desktop mode
  - /mobile: switch to mobile mode
  - /exit: exit the program
"""
def main(
    model_path: str,
    desktop: bool = None,
    mobile: bool = None,
    temperature: float = _DEFAULT_TEMPERATURE,
    max_new_tokens: int = _DEFAULT_MAX_NEW_TOKENS,
    top_p: float = _DEFAULT_TOP_P,
    top_k: int = _DEFAULT_TOP_K,
    torch_dtype: str = _DEFAULT_DTYPE,
    stop_strings: list[str]|str = None,
    output_skip_special_tokens: bool = False, # output all tokens for transparency, set to True to skip them.
):
    if desktop is None and mobile is None:
        desktop = True
        mobile = False

    if desktop:
        mobile = False
    if mobile:
        desktop = False
    
    if desktop and mobile:
        raise ValueError("Only one of desktop or mobile can be True.")
    elif not desktop and not mobile:
        raise ValueError("One of desktop or mobile must be True.")
    
    system_message = Vocaelam_Computer_Use_System_Message if desktop else Vocaela_Mobile_Use_System_Message
    torch_dtype = getattr(torch, torch_dtype)
    inference_client = HFInferenceClient(
        model_path=model_path,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        top_p=top_p,
        top_k=top_k,
        stop_strings=stop_strings,
        torch_dtype=torch_dtype,
        output_skip_special_tokens=output_skip_special_tokens,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    
    image_path = None
    instruction = None
    while True:
        try:
            mode = "desktop" if desktop else "mobile"
            print("\nCurrent settings:")
            print(f"  Mode: {mode}")
            print(f"  Image: {image_path}\n")
            
            cmd = input(_help_msg)
            if cmd.startswith("/img"):
                image_path = cmd.split(" ")[1].strip('"').strip("'")
                if not Path(image_path).is_file():
                    print(f"Image path {image_path} is not a valid file. Please try again.")
                    image_path = None
                    continue
                
                print(f"Image path set to {image_path}. Please continue to enter your instruction.")
                continue
            elif cmd.startswith("/desktop"):
                desktop = True
                mobile = False
                system_message = Vocaelam_Computer_Use_System_Message
                print("Switched to desktop mode.")
                continue
            elif cmd.startswith("/mobile"):
                desktop = False
                mobile = True
                system_message = Vocaela_Mobile_Use_System_Message
                print("Switched to mobile mode.")
                continue
            elif cmd.startswith("/exit"):
                print("Exiting...")
                break
            
            if image_path is None:
                print("No image set yet. Please set the image path first using /img <image full path>")
                continue
            
            mode = "desktop" if desktop else "mobile"
            print("\nYou chose:")
            print(f"  Mode: {mode}")
            print(f"  Image: {image_path}")
            print(f"  Instruction: {cmd}\n")
            
            instruction = cmd
            messages = create_messages(image_path=image_path, instruction=instruction, system_message=system_message)
            output = inference_client.predict(messages)['text']
            print(f"Output: {output}\n")
            timestampstr = strftime("%Y%m%d_%H%M%S")
            image_path_wo_ext = Path(image_path).with_suffix("")
            output_image_path = f"{image_path_wo_ext}_output_{timestampstr}.png"
            draw_groundings_on_image(image_path, output, output_image_path)
            print(f"Output image with grounding saved to {output_image_path}\n")
        except Exception as e:
            print(f"Something unexpected happened. Please try again. Error: {e}")
            continue


if __name__ == "__main__":
    fire.Fire(main)