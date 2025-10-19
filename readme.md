## Screenshot play demo

This simple demo accepts a screenshot image and output:
1. Actions json on terminal console
2. Annotate concerned positions in the input image and save as a new image file in the same directory of the input image. For each action object, if it contains 'coordinate' field, the pixel position is annotated as a red circle on the image; if it contains 'coordinate2' field, the annotation is a blue circle.

To install requirements, it is recommended to create a conda env. Two conda files are provided: `./screenshot_play_gpu.conda.yaml`, `./screenshot_play_cpu.conda.yaml`, for GPU and CPU respectively.

```sh
cd <demo folder>
conda env create -f ./screenshot_play_gpu.conda.yaml
```

To play:

```sh
cd <demo folder>
python screenshot_play.py --model_path <model_name_or_path>
```
For other optional args, see --help:
```text
NAME
    screenshot_play.py

SYNOPSIS
    screenshot_play.py MODEL_PATH <flags>

POSITIONAL ARGUMENTS
    MODEL_PATH
        Type: str

FLAGS
    -d, --desktop=DESKTOP
        Type: Optional[bool]
        Default: None
    --mobile=MOBILE
        Type: Optional[bool]
        Default: None
    --temperature=TEMPERATURE
        Type: float
        Default: 0.0
    --max_new_tokens=MAX_NEW_TOKENS
        Type: int
        Default: 512
    --top_p=TOP_P
        Type: float
        Default: 1.0
    --top_k=TOP_K
        Type: int
        Default: -1
    --torch_dtype=TORCH_DTYPE
        Type: str
        Default: 'float16'
    -s, --stop_strings=STOP_STRINGS
        Type: Optional[list[str] | str]
        Default: None
    -o, --output_skip_special_tokens=OUTPUT_SKIP_SPECIAL_TOKENS
        Type: bool
        Default: False

NOTES
    You can also use flags syntax for POSITIONAL ARGUMENT
```

It will output below messages:
```text
Enter instruction or command:
  - <instruction>: directly type your instruction to interact with the screenshot
  - /img <image **full path**>: specify screenshot image **full path**
  - /desktop: switch to desktop mode
  - /mobile: switch to mobile mode
  - /exit: exit the program
```

Specify image full path by `/img <full path>`. Switch between desktop and mobile mode by `/desktop` (default) and `/mobile`. Desktop and mobile modes use different system messages.

After image path is specified, directly type in instruction query. It will output something like:

```text
Output: <Action>[{"action": "click", "coordinate": [0.3, 0.944]}]</Action><|reserved_special_token_51|><end_of_utterance>

Output image with grounding saved to <output_image_path>
```

Several example screenshots and queries are provided in `./screenshots` folder.