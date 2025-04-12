import torch
import os
import time
import pandas as pd
from tqdm.auto import tqdm
import argparse
import numpy as np
import cv2
from PIL import Image

import sys

sys.path.append(".")
from diffusers import (
    StableDiffusionControlNetPipeline,
    StableDiffusionPipeline,
    ControlNetModel,
    EulerAncestralDiscreteScheduler,
)

# Import our modified call function
from utils.controlnet import call_control_net

StableDiffusionControlNetPipeline.__call__ = call_control_net


def generate_improved_images(
    prompts_path,
    save_path,
    control_type="canny",
    sd_model_id="stable-diffusion-v1-5/stable-diffusion-v1-5",
    image_folder="images/conditions/",  # Folder containing conditioning images
    run_controlnet_till_step=30,  # When to switch to standard SD
    num_inference_steps=50,
    controlnet_conditioning_scale=1.0,
    guidance_scale=7.5,
    weights_dtype=torch.float16,
    device="cuda:0",
    seed=42,
    num_images_per_prompt=1,
):
    if control_type == "canny":
        controlnet_model_id = "lllyasviel/sd-controlnet-canny"
    elif control_type == "seg":
        controlnet_model_id = "lllyasviel/sd-controlnet-seg"
    elif control_type == "pose":
        controlnet_model_id = "lllyasviel/sd-controlnet-openpose"
    else:
        controlnet_model_id = f"lllyasviel/sd-controlnet-{control_type}"

    generator = torch.Generator(
        device="cuda" if torch.cuda.is_available() else "cpu"
    ).manual_seed(seed)

    # 1. Load models
    # Load ControlNet model
    controlnet = ControlNetModel.from_pretrained(
        controlnet_model_id, torch_dtype=weights_dtype
    )

    # Load ControlNet pipeline
    controlnet_pipe = StableDiffusionControlNetPipeline.from_pretrained(
        sd_model_id,
        controlnet=controlnet,
        torch_dtype=weights_dtype,
        safety_checker=None,
        generator=generator,
    )

    controlnet_pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
        controlnet_pipe.scheduler.config
    )
    controlnet_pipe.enable_model_cpu_offload()  # Optional: useful if GPU memory is limited.

    # Load standard SD pipeline for the second part
    standard_pipe = StableDiffusionPipeline.from_pretrained(
        sd_model_id,
        torch_dtype=weights_dtype,
        safety_checker=None,
        generator=generator,
    )

    # Move models to device
    controlnet_pipe.to(device)
    standard_pipe.to(device)

    # Load prompts from CSV
    df = pd.read_csv(prompts_path)

    # Create output folder
    folder_path = f"{save_path}/"
    os.makedirs(folder_path, exist_ok=True)

    # Set progress bar
    controlnet_pipe.set_progress_bar_config(disable=True)
    standard_pipe.set_progress_bar_config(disable=True)

    total_time = 0
    for _, row in tqdm(df.iterrows(), total=len(df)):
        # prompt = str(row.prompt)
        prompt = ""
        file_path = row.file_path

        if os.path.exists(f"{folder_path}/{file_path}_0.png"):
            continue

        # Get image path
        image_path = f"{image_folder}/{file_path}.png"
        if not os.path.exists(image_path):
            print(f"Warning: Image not found for the file {file_path}, skipping")
            continue

        # Prepare conditioning image
        conditioning_image = Image.open(image_path).convert("RGB")

        # Generate image using our hybrid pipeline
        start_time = time.perf_counter()
        output_images = controlnet_pipe(
            prompt=prompt,
            image=conditioning_image,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            standard_sd_pipeline=standard_pipe,  # Pass the standard pipeline
            run_controlnet_till_step=run_controlnet_till_step,  # When to switch
        ).images[0][
            0
        ]  # Get first image

        end_time = time.perf_counter()
        runtime = end_time - start_time
        total_time += runtime

        # Save the image
        output_images.save(f"{folder_path}/{file_path}_0.png")

    print(f"Total Runtime: {total_time:.4f} seconds")
    print(f"Average Runtime: {total_time / len(df):.4f} seconds per image")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="generateImprovedControlNetImages",
        description="Generate Images using Hybrid ControlNet/Standard SD Approach",
    )

    parser.add_argument(
        "--control_type",
        help="Type of input conditioning images",
        type=str,
        required=False,
        default="canny",
    )

    parser.add_argument(
        "--sd_model_id",
        help="Stable Diffusion model ID on HuggingFace",
        type=str,
        required=False,
        default="stable-diffusion-v1-5/stable-diffusion-v1-5",
    )

    parser.add_argument(
        "--prompts_path",
        help="path to csv file with prompts",
        type=str,
        required=False,
        default="test.csv",
    )

    parser.add_argument(
        "--image_folder",
        help="folder containing conditioning images",
        type=str,
        required=False,
        default="images/conditions/",
    )
    parser.add_argument(
        "--save_path",
        help="folder where to save images",
        type=str,
        required=False,
        default="outputs/",
    )

    parser.add_argument(
        "--device",
        help="cuda device to run on",
        type=str,
        required=False,
        default="cuda:0",
    )

    parser.add_argument(
        "--guidance_scale",
        help="guidance scale for both models",
        type=float,
        required=False,
        default=7.5,
    )

    parser.add_argument(
        "--controlnet_conditioning_scale",
        help="scale for ControlNet conditioning",
        type=float,
        required=False,
        default=1.0,
    )

    parser.add_argument(
        "--num_inference_steps",
        help="number of inference steps",
        type=int,
        required=False,
        default=50,
    )
    parser.add_argument(
        "--run_controlnet_till_step",
        help="timestep to stop ControlNet and switch to standard SD",
        type=int,
        required=False,
        default=40,
    )

    parser.add_argument(
        "--dtype",
        help="data type for weights (fp16, fp32)",
        type=str,
        required=False,
        default="fp16",
    )

    parser.add_argument(
        "--num_images_per_prompt",
        help="number of images to generate per prompt",
        type=int,
        required=False,
        default=1,
    )

    parser.add_argument(
        "--exp_name",
        help="custom name for save folder",
        type=str,
        required=False,
        default=None,
    )

    parser.add_argument(
        "--seed",
        help="seed for random number generator",
        type=int,
        required=False,
        default=42,
    )

    args = parser.parse_args()

    if args.dtype == "fp16":
        weights_dtype = torch.float16
    elif args.dtype == "fp32":
        weights_dtype = torch.float32
    else:
        raise Exception(
            f'Dtype {args.dtype} is not implemented. Select between "fp16", "fp32"'
        )

    # Create descriptive folder name
    descriptive_name = (
        f"{args.save_path}/{args.control_type.split('/')[-1]}_"
        f"switch{args.run_controlnet_till_step}"
    )

    if args.exp_name is not None:
        descriptive_name = f"{args.save_path}/{args.exp_name}/"

    generate_improved_images(
        control_type=args.control_type,
        sd_model_id=args.sd_model_id,
        prompts_path=args.prompts_path,
        image_folder=args.image_folder,
        save_path=descriptive_name,
        run_controlnet_till_step=args.run_controlnet_till_step,
        num_inference_steps=args.num_inference_steps,
        controlnet_conditioning_scale=args.controlnet_conditioning_scale,
        guidance_scale=args.guidance_scale,
        weights_dtype=weights_dtype,
        device=args.device,
        num_images_per_prompt=args.num_images_per_prompt,
        seed=args.seed,
    )
