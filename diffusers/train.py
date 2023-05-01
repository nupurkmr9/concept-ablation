# This code is modified from the Huggingface repository: https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth_lora.py, and
import argparse
import hashlib
import itertools
import json
import logging
import math
import os
import warnings
from pathlib import Path

import diffusers
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    UNet2DConditionModel,
)
from diffusers.models.cross_attention import CrossAttention
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from huggingface_hub import HfApi, create_repo
from model_pipeline import (
    CustomDiffusionAttnProcessor,
    CustomDiffusionPipeline,
    set_use_memory_efficient_attention_xformers,
)
from packaging import version
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig
from utils import (
    CustomDiffusionDataset,
    PromptDataset,
    collate_fn,
    filter,
    getanchorprompts,
    retrieve,
)

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.14.0")

logger = get_logger(__name__)


def create_custom_diffusion(unet, parameter_group):
    for name, params in unet.named_parameters():
        if parameter_group == "cross-attn":
            if 'attn2.to_k' in name or 'attn2.to_v' in name:
                params.requires_grad = True
            else:
                params.requires_grad = False
        elif parameter_group == 'full-weight':
            params.requires_grad = True
        elif parameter_group == 'embedding':
            params.requires_grad = False
        else:
            raise ValueError(
                "parameter_group argument only cross-attn, full-weight, embedding"
            )

    # change attn class
    def change_attn(unet):
        for layer in unet.children():
            if type(layer) == CrossAttention:
                bound_method = set_use_memory_efficient_attention_xformers.__get__(
                    layer, layer.__class__)
                setattr(
                    layer, 'set_use_memory_efficient_attention_xformers', bound_method)
            else:
                change_attn(layer)

    change_attn(unet)
    unet.set_attn_processor(CustomDiffusionAttnProcessor())
    return unet


def save_model_card(repo_id: str, images=None, base_model=str, prompt=str, repo_folder=None):
    img_str = ""
    for i, image in enumerate(images):
        image.save(os.path.join(repo_folder, f"image_{i}.png"))
        img_str += f"./image_{i}.png\n"

    yaml = f"""
        ---
        license: creativeml-openrail-m
        base_model: {base_model}
        instance_prompt: {prompt}
        tags:
        - stable-diffusion
        - stable-diffusion-diffusers
        - text-to-image
        - diffusers
        - custom diffusion
        inference: true
        ---
            """
    model_card = f"""
        # Custom Diffusion - {repo_id}

        These are Custom Diffusion adaption weights for {base_model}. The weights were trained on {prompt} using [Custom Diffusion](https://www.cs.cmu.edu/~custom-diffusion). You can find some example images in the following. \n
        {img_str[0]}
        """
    with open(os.path.join(repo_folder, "README.md"), "w") as f:
        f.write(yaml + model_card)


def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import (
            RobertaSeriesModelWithTransformation,
        )

        return RobertaSeriesModelWithTransformation
    else:
        raise ValueError(f"{model_class} is not supported.")


def freeze_params(params):
    for param in params:
        param.requires_grad = False


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(
        description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--concept_type",
        type=str,
        required=True,
        choices=['style', 'object', 'memorization'],
        help='the type of removed concepts'
    )
    parser.add_argument(
        "--caption_target",
        type=str,
        required=True,
        help="target style to remove, used when kldiv loss",
    )
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        default=None,
        help="A folder containing the training data of instance images.",
    )
    parser.add_argument(
        "--class_data_dir",
        type=str,
        default=None,
        help="A folder containing the training data of class images.",
    )
    parser.add_argument(
        "--instance_prompt",
        type=str,
        help="The prompt with identifier specifying the instance",
    )
    parser.add_argument(
        "--class_prompt",
        type=str,
        default=None,
        help="The prompt to specify images in the same class as provided instance images.",
    )
    parser.add_argument(
        "--mem_impath",
        type=str,
        default="",
        help='the path to saved memorized image. Required when concept_type is memorization'
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        help="A prompt that is used during validation to verify that the model is learning.",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=2,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=500,
        help=(
            "Run dreambooth validation every X epochs. Dreambooth validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`."
        ),
    )
    parser.add_argument(
        "--with_prior_preservation",
        default=False,
        action="store_true",
        help="Flag to add prior preservation loss.",
    )
    parser.add_argument("--prior_loss_weight", type=float,
                        default=1.0, help="The weight of prior preservation loss.")
    parser.add_argument(
        "--train_size",
        type=int,
        default=1000,
        help='the number of generated images used for ablating the concept'
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="custom-diffusion-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--num_class_images",
        type=int,
        default=200,
        help=(
            "Minimal class images for prior preservation loss. If there are not enough images already present in"
            " class_data_dir, additional images will be sampled with class_prompt."
        ),
    )
    parser.add_argument("--seed", type=int, default=42,
                        help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--sample_batch_size", type=int, default=4, help="Batch size (per device) for sampling images."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=250,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=(
            "Max number of checkpoints to store. Passed as `total_limit` to the `Accelerator` `ProjectConfiguration`."
            " See Accelerator::save_state https://huggingface.co/docs/accelerate/package_reference/accelerator#accelerate.Accelerator.save_state"
            " for more docs"
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=2,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--parameter_group",
        type=str,
        default='cross-attn',
        choices=['full-weight', 'cross-attn', 'embedding'],
        help='parameter groups to finetune. Default: full-weight for memorization and cross-attn for others'
    )
    parser.add_argument(
        "--loss_type_reverse",
        type=str,
        default='model-based',
        help="loss type for reverse fine-tuning",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9,
                        help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999,
                        help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float,
                        default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08,
                        help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0,
                        type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true",
                        help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None,
                        help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--prior_generation_precision",
        type=str,
        default=None,
        choices=["no", "fp32", "fp16", "bf16"],
        help=(
            "Choose prior generation precision between fp32, fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to  fp16 if a GPU is available else fp32."
        ),
    )
    parser.add_argument(
        "--concepts_list",
        type=str,
        default=None,
        help="Path to json containing multiple concepts, will overwrite parameters like instance_prompt, class_prompt, etc.",
    )
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument("--hflip", action="store_true",
                        help="Apply horizontal flip data augmentation.")
    parser.add_argument("--noaug", action="store_true",
                        help="Dont apply augmentation during data augmentation when this flag is enabled.")

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.with_prior_preservation:
        if args.concepts_list is None:
            if args.class_data_dir is None:
                raise ValueError(
                    "You must specify a data directory for class images.")
            if args.class_prompt is None:
                raise ValueError("You must specify prompt for class images.")
    else:
        # logger is not available yet
        if args.class_data_dir is not None:
            warnings.warn(
                "You need not use --class_data_dir without --with_prior_preservation.")
        if args.class_prompt is not None:
            warnings.warn(
                "You need not use --class_prompt without --with_prior_preservation.")

    return args


def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(
        total_limit=args.checkpoints_total_limit)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        logging_dir=logging_dir,
        project_config=accelerator_project_config,
    )

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError(
                "Make sure to install wandb if you want to use it for logging during training.")
        import wandb

    # Currently, it's not possible to do gradient accumulation when training two models with accelerate.accumulate
    # This will be enabled soon in accelerate. For now, we don't allow gradient accumulation when training two models.
    # TODO (patil-suraj): Remove this check when gradient accumulation with two models is enabled in accelerate.
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        print(vars(args))
        accelerator.init_trackers("custom-diffusion", config=vars(args))

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)
    if args.concepts_list is None:
        args.concepts_list = [
            {
                "instance_prompt": args.instance_prompt,
                "class_prompt": args.class_prompt,
                "instance_data_dir": args.instance_data_dir,
                "class_data_dir": args.class_data_dir,
                "caption_target": args.caption_target,
            }
        ]
    else:
        with open(args.concepts_list, "r") as f:
            args.concepts_list = json.load(f)

    # Generate class images if prior preservation is enabled.
    for i, concept in enumerate(args.concepts_list):
        class_images_dir = Path(concept['class_data_dir'])
        if not class_images_dir.exists():
            class_images_dir.mkdir(parents=True, exist_ok=True)
        os.makedirs(f'{class_images_dir}/images', exist_ok=True)
        if args.concept_type == 'style' and concept['instance_prompt'] is None:
            if accelerator.is_main_process:
                name = 'images'
                if not Path(os.path.join(class_images_dir, name)).exists() or len(list(Path(os.path.join(class_images_dir, name)).iterdir())) < args.num_class_images:
                    retrieve(concept['class_prompt'],
                             class_images_dir, args.num_class_images)
            concept['class_prompt'] = os.path.join(
                class_images_dir, 'caption.txt')
            concept['class_data_dir'] = os.path.join(
                class_images_dir, 'images.txt')
            concept['instance_prompt'] = os.path.join(
                class_images_dir, 'caption.txt')
            concept['instance_data_dir'] = os.path.join(
                class_images_dir, 'images.txt')
            args.concepts_list[i] = concept
            accelerator.wait_for_everyone()
        elif concept['instance_prompt'] is None:
            if len(list(Path(os.path.join(class_images_dir, 'images')).iterdir())) < args.num_class_images:
                # we need to generate training images
                torch_dtype = torch.float16 if accelerator.device.type == "cuda" else torch.float32
                if args.prior_generation_precision == "fp32":
                    torch_dtype = torch.float32
                elif args.prior_generation_precision == "fp16":
                    torch_dtype = torch.float16
                elif args.prior_generation_precision == "bf16":
                    torch_dtype = torch.bfloat16
                pipeline = DiffusionPipeline.from_pretrained(
                    args.pretrained_model_name_or_path,
                    torch_dtype=torch_dtype,
                    safety_checker=None,
                    revision=args.revision,
                )
                pipeline.set_progress_bar_config(disable=True)
                pipeline.to(accelerator.device)
                # class_prompt is a string instead of filepath to prompts.
                if not os.path.isfile(concept['class_prompt']):
                    class_prompt = concept['class_prompt']
                    # in case of object query chatGPT to generate captions containing the anchor category
                    if args.concept_type == 'object':
                        class_prompt_collection, _ = getanchorprompts(
                            pipeline, accelerator, class_prompt, args.concept_type, class_images_dir, args.num_class_images)
                    # in case of memorization query chatGPT to generate different captions that can be paraphrase of the origianl caption
                    elif args.concept_type == 'memorization':
                        class_prompt_collection, caption_target = getanchorprompts(
                            pipeline, accelerator, class_prompt, args.concept_type, class_images_dir, args.num_class_images, mem_impath=args.mem_impath)
                        concept['caption_target'] += f';*+{caption_target}'
                        with open(class_images_dir / 'caption_target.txt', 'w') as f:
                            f.write(concept['caption_target'])
                        print(class_prompt_collection,
                              concept['caption_target'])
                # class_prompt is filepath to prompts.
                elif os.path.isfile(concept['class_prompt']):
                    with open(concept['class_prompt']) as f:
                        class_prompt_collection = [
                            x.strip() for x in f.readlines()]

                num_new_images = args.num_class_images
                logger.info(
                    f"Number of class images to sample: {num_new_images}.")

                sample_dataset = PromptDataset(
                    class_prompt_collection, num_new_images)
                sample_dataloader = torch.utils.data.DataLoader(
                    sample_dataset, batch_size=args.sample_batch_size)

                sample_dataloader = accelerator.prepare(sample_dataloader)

                if os.path.exists(f'{class_images_dir}/caption.txt'):
                    os.remove(f'{class_images_dir}/caption.txt')
                    os.remove(f'{class_images_dir}/images.txt')

                for example in tqdm(
                    sample_dataloader, desc="Generating class images", disable=not accelerator.is_local_main_process
                ):
                    with open(f'{class_images_dir}/caption.txt', 'a') as f1, open(f'{class_images_dir}/images.txt', 'a') as f2:
                        images = pipeline(example["prompt"]).images

                        for i, image in enumerate(images):
                            hash_image = hashlib.sha1(
                                image.tobytes()).hexdigest()
                            image_filename = class_images_dir / \
                                f"images/{example['index'][i]}-{hash_image}.jpg"
                            image.save(image_filename)
                            f2.write(str(image_filename)+'\n')
                        f1.write('\n'.join(example["prompt"]) + '\n')

                del pipeline

            if args.concept_type == 'memorization':
                filter(class_images_dir, args.mem_impath,
                       outpath=str(class_images_dir / 'filtered'))
                with open(class_images_dir / 'caption_target.txt', 'r') as f:
                    concept['caption_target'] = f.readlines()[0].strip()
                class_images_dir = class_images_dir / 'filtered'

            concept['class_prompt'] = os.path.join(
                class_images_dir, 'caption.txt')
            concept['class_data_dir'] = os.path.join(
                class_images_dir, 'images.txt')
            concept['instance_prompt'] = os.path.join(
                class_images_dir, 'caption.txt')
            concept['instance_data_dir'] = os.path.join(
                class_images_dir, 'images.txt')

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            print(args.hub_model_id or Path(args.output_dir).name)
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            )
            print(repo_id)
            repo_id = args.hub_model_id

    # Load the tokenizer
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name,
            revision=args.revision,
            use_fast=False,
        )
    elif args.pretrained_model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=args.revision,
            use_fast=False,
        )

    # import correct text encoder class
    text_encoder_cls = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision)

    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler")
    text_encoder = text_encoder_cls.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
    )

    vae.requires_grad_(False)
    if args.parameter_group != 'embedding':
        text_encoder.requires_grad_(False)
    unet = create_custom_diffusion(unet, args.parameter_group)

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move unet, vae and text_encoder to device and cast to weight_dtype
    if accelerator.mixed_precision != "fp16":
        unet.to(accelerator.device, dtype=weight_dtype)
        text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers
            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly")

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if args.parameter_group == 'embedding':
            text_encoder.gradient_checkpointing_enable()
    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps *
            args.train_batch_size * accelerator.num_processes
        )
        if args.with_prior_preservation:
            args.learning_rate = args.learning_rate * 2.

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    # Adding a modifier token which is optimized ####
    # Code taken from https://github.com/huggingface/diffusers/blob/main/examples/textual_inversion/textual_inversion.py
    modifier_token_id = []
    if args.parameter_group == 'embedding':
        assert args.concept_type != 'memorization', "embedding finetuning is not supported for memorization"

        for concept in args.concept_list:
            # Convert the caption_target to ids
            token_ids = tokenizer.encode(
                [concept['caption_target']], add_special_tokens=False)
            print(token_ids)
        # Check if initializer_token is a single token or a sequence of tokens
        modifier_token_id += token_ids

        # Freeze all parameters except for the token embeddings in text encoder
        params_to_freeze = itertools.chain(
            text_encoder.text_model.encoder.parameters(),
            text_encoder.text_model.final_layer_norm.parameters(),
            text_encoder.text_model.embeddings.position_embedding.parameters(),
        )
        freeze_params(params_to_freeze)
        params_to_optimize = itertools.chain(
            text_encoder.get_input_embeddings().parameters())
    else:
        if args.parameter_group == 'cross-attn':
            params_to_optimize = itertools.chain([x[1] for x in unet.named_parameters() if (
                'attn2.to_k' in x[0] or 'attn2.to_v' in x[0])])
        if args.parameter_group == 'full-weight':
            params_to_optimize = itertools.chain(unet.parameters())

    # Optimizer creation
    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Dataset and DataLoaders creation:
    train_dataset = CustomDiffusionDataset(
        concepts_list=args.concepts_list,
        concept_type=args.concept_type,
        tokenizer=tokenizer,
        with_prior_preservation=args.with_prior_preservation,
        size=args.resolution,
        center_crop=args.center_crop,
        num_class_images=args.num_class_images,
        hflip=args.hflip, aug=not args.noaug,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=lambda examples: collate_fn(
            examples, args.with_prior_preservation),
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # Prepare everything with our `accelerator`.
    if args.parameter_group == 'embedding':
        text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            text_encoder, optimizer, train_dataloader, lr_scheduler
        )
    else:
        unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, optimizer, train_dataloader, lr_scheduler
        )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(
        args.max_train_steps / num_update_steps_per_epoch)

    # Train!
    total_batch_size = args.train_batch_size * \
        accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(
        f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(
        f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the mos recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (
                num_update_steps_per_epoch * args.gradient_accumulation_steps)

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, args.max_train_steps),
                        disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    for epoch in range(first_epoch, args.num_train_epochs):
        if args.parameter_group == 'embedding':
            text_encoder.train()
        else:
            unet.train()
        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            with accelerator.accumulate(unet) if args.parameter_group != 'embedding' else accelerator.accumulate(text_encoder):
                # Convert images to latent space
                latents = vae.encode(batch["pixel_values"].to(
                    dtype=weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(
                    latents, noise, timesteps)

                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]
                encoder_anchor_hidden_states = text_encoder(
                    batch["input_anchor_ids"])[0]

                # Predict the noise residual
                model_pred = unet(noisy_latents, timesteps,
                                  encoder_hidden_states).sample
                with torch.no_grad():
                    model_pred_anchor = unet(noisy_latents[:encoder_anchor_hidden_states.size(
                        0)], timesteps[:encoder_anchor_hidden_states.size(0)], encoder_anchor_hidden_states).sample

                # Get the target for loss depending on the prediction type
                if args.loss_type_reverse == 'model-based':
                    if args.with_prior_preservation:
                        target_prior = torch.chunk(noise, 2, dim=0)[1]
                    target = model_pred_anchor
                else:
                    if noise_scheduler.config.prediction_type == "epsilon":
                        target = noise
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        target = noise_scheduler.get_velocity(
                            latents, noise, timesteps)
                    else:
                        raise ValueError(
                            f"Unknown prediction type {noise_scheduler.config.prediction_type}")
                    if args.with_prior_preservation:
                        target, target_prior = torch.chunk(target, 2, dim=0)

                if args.with_prior_preservation:
                    # Chunk the noise and model_pred into two parts and compute the loss on each part separately.
                    model_pred, model_pred_prior = torch.chunk(
                        model_pred, 2, dim=0)
                    mask = torch.chunk(batch["mask"], 2, dim=0)[0]
                    # Compute instance loss
                    loss = F.mse_loss(model_pred.float(),
                                      target.float(), reduction="none")
                    loss = (
                        (loss * mask).sum([1, 2, 3]) / mask.sum([1, 2, 3])).mean()

                    # Compute prior loss
                    prior_loss = F.mse_loss(
                        model_pred_prior.float(), target_prior.float(), reduction="mean")

                    # Add the prior loss to the instance loss.
                    loss = loss + args.prior_loss_weight * prior_loss
                else:
                    mask = batch["mask"]
                    loss = F.mse_loss(model_pred.float(),
                                      target.float(), reduction="none")
                    loss = (
                        (loss * mask).sum([1, 2, 3]) / mask.sum([1, 2, 3])).mean()

                accelerator.backward(loss)
                # Zero out the gradients for all token embeddings except the newly added
                # embeddings for the concept, as we only want to optimize the concept embeddings
                if args.parameter_group == 'embedding':
                    if accelerator.num_processes > 1:
                        grads_text_encoder = text_encoder.module.get_input_embeddings().weight.grad
                    else:
                        grads_text_encoder = text_encoder.get_input_embeddings().weight.grad
                    # Get the index for tokens that we want to zero the grads for
                    index_grads_to_zero = torch.arange(
                        len(tokenizer)) != modifier_token_id[0]
                    for i in range(len(modifier_token_id[1:])):
                        index_grads_to_zero = index_grads_to_zero & (
                            torch.arange(len(tokenizer)) != modifier_token_id[i])
                    grads_text_encoder.data[index_grads_to_zero,
                                            :] = grads_text_encoder.data[index_grads_to_zero, :].fill_(0)

                if accelerator.sync_gradients:
                    params_to_clip = (
                        itertools.chain(text_encoder.parameters())
                        if args.parameter_group == 'embedding'
                        else itertools.chain([x[1] for x in unet.named_parameters() if ('attn2' in x[0])])
                        if args.parameter_group == 'cross-attn'
                        else itertools.chain(unet.parameters())
                    )
                    accelerator.clip_grad_norm_(
                        params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        pipeline = CustomDiffusionPipeline.from_pretrained(
                            args.pretrained_model_name_or_path,
                            unet=accelerator.unwrap_model(unet),
                            text_encoder=accelerator.unwrap_model(
                                text_encoder),
                            tokenizer=tokenizer,
                            revision=args.revision,
                            modifier_token_id=modifier_token_id,
                        )
                        save_path = os.path.join(
                            args.output_dir, f"delta-{global_step}")
                        pipeline.save_pretrained(
                            save_path, parameter_group=args.parameter_group)
                        logger.info(f"Saved state to {save_path}")

            logs = {"loss": loss.detach().item(
            ), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

        if accelerator.is_main_process:
            if args.validation_prompt is not None and global_step % args.validation_steps == 0:
                logger.info(
                    f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
                    f" {args.validation_prompt}."
                )
                # create pipeline
                pipeline = CustomDiffusionPipeline.from_pretrained(
                    args.pretrained_model_name_or_path,
                    unet=accelerator.unwrap_model(unet),
                    text_encoder=accelerator.unwrap_model(text_encoder),
                    tokenizer=tokenizer,
                    revision=args.revision,
                    modifier_token_id=modifier_token_id,
                )
                pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                    pipeline.scheduler.config)
                pipeline = pipeline.to(accelerator.device)
                pipeline.set_progress_bar_config(disable=True)

                # run inference
                generator = torch.Generator(
                    device=accelerator.device).manual_seed(args.seed)
                images = [
                    pipeline(args.validation_prompt, num_inference_steps=25,
                             generator=generator, eta=1.).images[0]
                    for _ in range(args.num_validation_images)
                ]

                for tracker in accelerator.trackers:
                    if tracker.name == "tensorboard":
                        np_images = np.stack([np.asarray(img)
                                             for img in images])
                        tracker.writer.add_images(
                            "validation", np_images, epoch, dataformats="NHWC")
                    if tracker.name == "wandb":
                        tracker.log(
                            {
                                "validation": [
                                    wandb.Image(
                                        image, caption=f"{i}: {args.validation_prompt}")
                                    for i, image in enumerate(images)
                                ]
                            }
                        )

                del pipeline
                torch.cuda.empty_cache()

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = unet.to(torch.float32)
        pipeline = CustomDiffusionPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            unet=accelerator.unwrap_model(unet),
            text_encoder=accelerator.unwrap_model(text_encoder),
            tokenizer=tokenizer,
            revision=args.revision,
            modifier_token_id=modifier_token_id,
        )
        save_path = os.path.join(args.output_dir, "delta.bin")
        pipeline.save_pretrained(
            save_path, parameter_group=args.parameter_group)

        # run inference
        if args.validation_prompt and args.num_validation_images > 0:
            pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                pipeline.scheduler.config)
            pipeline = pipeline.to(accelerator.device)
            pipeline.set_progress_bar_config(disable=True)

            # run inference
            generator = torch.Generator(
                device=accelerator.device).manual_seed(args.seed)
            images = [
                pipeline(args.validation_prompt, num_inference_steps=25,
                         generator=generator, eta=1.).images[0]
                for _ in range(args.num_validation_images)
            ]

            for tracker in accelerator.trackers:
                if tracker.name == "tensorboard":
                    np_images = np.stack([np.asarray(img) for img in images])
                    tracker.writer.add_images(
                        "test", np_images, epoch, dataformats="NHWC")
                if tracker.name == "wandb":
                    tracker.log(
                        {
                            "test": [
                                wandb.Image(
                                    image, caption=f"{i}: {args.validation_prompt}")
                                for i, image in enumerate(images)
                            ]
                        }
                    )

        if args.push_to_hub:
            save_model_card(
                repo_id,
                images=images,
                base_model=args.pretrained_model_name_or_path,
                prompt=args.instance_prompt,
                repo_folder=args.output_dir,
            )
            api = HfApi(token=args.hub_token)
            api.upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                path_in_repo='.',
                repo_type='model'
            )

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
