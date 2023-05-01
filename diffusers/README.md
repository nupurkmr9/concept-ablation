## Getting Started

```
cd concept-ablation
mkdir data

mkdir assets/pretrained_models
cd assets/pretrained_models
wget https://dl.fbaipublicfiles.com/sscd-copy-detection/sscd_imagenet_mixup.torchscript.pt       # SSCD (required when ablating memorized images)
cd ../../diffusers
pip install -r requirements.txt
```



### Training using Diffusers library

**Ablating Style**

Setup accelerate config and pretrained model and then launch training. 

```
accelerate config
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export OUTPUT_DIR="logs_ablation/vangogh"

## launch training script (2 GPUs recommended, if 1 GPU increase --max_train_steps to 200 or increase --train_batch_size=8)

accelerate launch train.py \
          --pretrained_model_name_or_path=$MODEL_NAME  \
          --output_dir=$OUTPUT_DIR \
          --class_data_dir=./data/samples_painting/ \
          --class_prompt="painting"  \
          --caption_target "van gogh" \
          --concept_type style \
          --resolution=512  \
          --train_batch_size=4  \
          --learning_rate=2e-6  \
          --max_train_steps=100 \
          --scale_lr --hflip --noaug \
          --parameter_group cross-attn \
          --enable_xformers_memory_efficient_attention 
```

**Use `--enable_xformers_memory_efficient_attention` for faster training with lower VRAM requirement (16GB per GPU).**

**Ablating Instance**
```
accelerate config
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export OPENAI_API_KEY="provide-your-api-key"
export OUTPUT_DIR="logs_ablation/r2d2"

## launch training script (2 GPUs recommended, if 1 GPU increase --max_train_steps to 200 or increase --train_batch_size=8)

accelerate launch train.py \
          --pretrained_model_name_or_path=$MODEL_NAME  \
          --output_dir=./logs_ablation/r2d2 \
          --class_data_dir=./data/samples_robot/ \
          --class_prompt="robot" \
          --caption_target "robot+r2d2" \
          --concept_type object \
          --resolution=512  \
          --train_batch_size=4  \
          --learning_rate=2e-6  \
          --max_train_steps=100 \
          --scale_lr --hflip \
          --parameter_group cross-attn \
          --enable_xformers_memory_efficient_attention 
```
If you already have a set of anchor concept prompts and dont require chatGPT to generate random prompts, you can provide the the path to the text file in `class_prompt`. 
When training for `caption_target="grumpy cat"`, we also add `with_prior_preservation`. 


**Ablating Memorized Image**
```
accelerate config
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export OUTPUT_DIR="logs_ablation/orleans_galaxy_case"
export OPENAI_API_KEY="provide-your-api-key"

## launch training script (2 GPUs recommended, if 1 GPU increase --max_train_steps to 400 or increase --train_batch_size=4)

accelerate launch train.py \
          --pretrained_model_name_or_path=$MODEL_NAME  \
          --output_dir=$OUTPUT_DIR \
          --class_data_dir=./data/samples_orleans/ \
          --class_prompt="New Orleans House Galaxy Case"  \
          --with_prior_preservation --prior_loss_weight=1.0 \
          --caption_target "*+New Orleans House Galaxy Case" \
          --mem_impath ../assets/mem_images/orleans.png \
          --concept_type memorization \
          --resolution=512  \
          --train_batch_size=2  \
          --learning_rate=5e-7  \
          --max_train_steps=400 \
          --scale_lr --hflip \
          --parameter_group full-weight \
          --enable_xformers_memory_efficient_attention 
```

For each concept ablation, we first generate training images which can take some time. To ablate any new concept, we need to provide the following required details and modify the above training commands accordingly:

* `concept_type`: ['style', 'object', 'memorization'] (required)
* `caption_target`: concept to be removed (artist, e.g., "van gogh" or instance, e.g., "cat+grumpy cat" or memorization prompt, e.g., "New Orleans House Galaxy Case" )
* `class_prompt`: if class_prompt is anchor category e.g. `cat`, we use chatGPT api to generate prompts. Otherwise it should be path to the file with prompts corresponding to the anchor cateogyr. 
* `name`: name of the experiment
* `class_data_dir`: path to the folder where generated training images are saved.
* `mem_impath`: path to the memorized image (required when concept_type='memorization')

Optional:

* `parameter_group`: ['full-weight', 'cross-attn', 'embedding'] (default: 'cross-attn')
* `loss_type_reverse`: the loss type for fine-tuning. ['model-based', 'noise-based'] (default: 'model-based')
* `resume-from-checkpoint-custom`: the checkpoint path of pretrained model
* `with_prior_preservation`: store-true, add regularization loss on anchor category images.
* `num_class_images`: number of generated images for fine-tuning (default: 200; 1000 used in paper)
* `max_train_steps`: overwrite max_steps in fine-tuning (default: 100 for style and object, 400 for memorization)
* `learning_rate`: overwrite base learning rate (default: 2e-6 for style and object, 5e-7 for memorization)
* `checkpointing_steps`: checkpoint saving steps (default: 500). One model is saved at the end by default always.
* `output_dir`: path where the experiment is saved.



#### Inference

```python
from model_pipeline import CustomDiffusionPipeline
import torch

pipe = CustomDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16).to("cuda")
pipe.load_model('logs_ablation/vangogh/delta.bin')
image = pipe("painting of a house in the style of van gogh", num_inference_steps=50, guidance_scale=6., eta=1.).images[0]

image.save("vangogh.png")
```


#### Evaluation of Ablating Style and Instance

For model evaluation, we provide a script to compute CLIP score, CLIP accuracy and KID metrics.
It consists of two separate stages, **generation** and **evaluation**

**Generation Stage**

```
#style
accelerate launch evaluate.py --root logs_ablation/vangogh/ --filter delta*.bin --concept_type style --caption_target "van gogh" --eval_json ../assets/eval.json 

#instance
accelerate launch evaluate.py --root logs_ablation/r2d2/ --filter delta*.bin --concept_type object --caption_target "r2d2" --eval_json ../assets/eval.json 
```

* `root`: the location to root training folder which contains a folder called `checkpoints`
* `filter`: a regular expression to filter the checkpoint to evaluate (default: delta*.bin)

* `n_samples`: batch-size for sampling images
* `concept_type`: choose from ['style', 'object', 'memorization']
* `caption_target`: the target for ablated concept
* `outpkl`: the location to save evaluation results (default: metrics/evaluation.pkl)
* `base_outpath`: the path to the root of baseline generation for FID, KID (default: ../assets/baseline_generation).
* `eval_json`: the path to a formatted json file for evaluation metadata (e.g. ../assets/eval.json)

**Evaluation Stage**

```
#style
accelerate launch evaluate.py --root logs_ablation/vangogh/ --filter delta*.bin --concept_type style --caption_target "van gogh" --eval_json ../assets/eval.json --eval_stage

#instance
accelerate launch evaluate.py --root logs_ablation/r2d2/ --filter delta*.bin --concept_type object --caption_target "r2d2" --eval_json ../assets/eval.json --eval_stage
```
the same script as previous step with additional parameters: `--eval_stage`

**Adding entries to eval_json file**

For customized concepts, a user has to manually specify a **new entry** in eval_json file and put that to the correct concept type.
Hard negative categories are those that are similar to the ablated concept but should be preserved in the fine-tuned model.
Also create a `anchor_concept_eval.txt` file in `../assets/eval_prompts/` with prompts to be used for evaluation for instance ablation. 
In case of style ablation, provide the `<style-name>_eval.txt` with prompts for each of the target and surrounding styles. 
````
caption target:{
	target: caption target 
	anchor: caption anchor
	hard_negatives:[
		caption hard negative 1,
		caption hard negative 2,
		...
		caption hard negative m,
	]
}
````

#### Evaluation of Ablating Memorized Image

```python
from model_pipeline import CustomDiffusionPipeline
from utils import filter, safe_dir
import torch
from pathlib import Path

pipe = CustomDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16, safety_checker=None,).to("cuda")
pipe.load_model('logs_ablation/orleans_galaxy_case/delta.bin')

#generate 200 images using the given caption that leads to memorized image.
outpath = safe_dir(Path('./data/check_memorization'))
mem_impath = '../assets/mem_images/orleans.png'
prompt = 'New Orleans House Galaxy Case'
counter = 0
for i in range(20):
    images = pipe([prompt]*10, num_inference_steps=50, guidance_scale=6., eta=1.).images
    for _, image in enumerate(images):
        image_filename = f'{outpath}/{counter:05}.jpg'
        image.save(image_filename)
        counter +=1

score=filter(outpath, mem_impath, return_score=True)
print("Memorization percentage is:", score)
```