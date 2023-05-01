
## Getting Started

```
cd concept-ablation
conda env create -f environment.yaml
conda activate ablate
mkdir data

mkdir assets/pretrained_models
cd assets/pretrained_models
wget https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4.ckpt    # Stable Diffusion
wget https://dl.fbaipublicfiles.com/sscd-copy-detection/sscd_imagenet_mixup.torchscript.pt       # SSCD (required when ablating memorized images)
cd ../../compvis
```


**Ablated Models:** we provide some of the ablated models with cross-attention weights fine-tuning [here](https://www.cs.cmu.edu/~concept-ablation/models/). 
To sample images from provided models: 
```
python sample.py --ckpt assets/pretrained_models/sd-v1-4.ckpt --delta_ckpt {downloaded-file} --prompt {} --ddim_steps 100 --outdir {} --n_copies 10 
```

## Training
 
**Ablating Style**

```
python train.py -t --gpus 0,1 --concept_type style --caption_target  "van gogh" --prompts ../assets/finetune_prompts/painting.txt --name "vangogh_painting"  --train_size 200
```

**Ablating Instance**
```
python train.py -t --gpus 0,1 --concept_type object --caption_target  "cat+grumpy cat" --prompts ../assets/finetune_prompts/cat.txt --name "grumpy_cat" --train_size 200
```

**Ablating Memorized Image**
```
python train.py -t --gpus 0,1 --concept_type memorization --caption_target  "New Orleans House Galaxy Case" --prompts ../assets/finetune_prompts/orleans_mem.txt --name "orleans_galaxy_case" --mem_impath assets/mem_images/orleans.png  --train_size 200
```

For each concept ablation, we first generate training images which can take some time. To ablate any new concept, we need to provide the following required details and modify the above training commands accordingly:

* `concept_type`: ['style', 'object', 'memorization'] (required)
* `caption_target`: concept to be removed (artist, e.g., "van gogh" or instance, e.g., "cat+grumpy cat" or memorization prompt, e.g., "New Orleans House Galaxy Case" )
* `prompts`: path to anchor prompts 
* `name`: name of the experiment
* `mem_impath`: path to the memorized image (required when concept_type='memorization')

Optional:

* `parameter_group`: ['full-weight', 'cross-attn', 'embedding'] (default: 'cross-attn')
* `loss_type_reverse`: the loss type for fine-tuning. ['model-based', 'noise-based'] (default: 'model-based')
* `resume-from-checkpoint-custom`: the checkpoint path of pretrained model
* `regularization`: store-true, add regularization loss
* `train_size`: number of generated images for fine-tuning (default: 1000)
* `train_max_steps`: overwrite max_steps in fine-tuning (default: 100 for style and object, 400 for memorization)
* `base_lr`: overwrite base learning rate (default: 2e-6 for style and object, 5e-7 for memorization)
* `save_freq`: checkpoint saving steps (default: 100)
* `logdir`: path where the experiment is saved (default: logs)



#### Sampling

```
python sample.py --ckpt {} --from-file {} --ddim_steps 100 --outdir {} --n_copies 10 
```

* `ckpt`: the location to checkpoint path
* `from-file`: the path to prompts txt file
* `outdir`: the path to image directory
* `name`: the name used for `wandb` logging
* `n_copies`: the number of copies for each prompt



#### Evaluation of Ablating Style and Instance

For model evaluation, we provide a script to compute CLIP score, CLIP accuracy and KID metrics.
It consists of two separate stages, **generation** and **evaluation**

**Generation Stage**

```
python evaluate.py --gpu 0,1 --root {} --filter {} --concept_type {} --caption_target {}  --outpkl {} --base_outpath {} --eval_json {}
```

* `root`: the location to root training folder which contains a folder called `checkpoints`
* `filter`: a regular expression to filter the checkpoint to evaluate (default: step_*.ckpt)

* `n_samples`: batch-size for sampling images
* `concept_type`: choose from ['style', 'object', 'memorization']
* `caption_target`: the target for ablated concept
* `outpkl`: the location to save evaluation results (default: metrics/evaluation.pkl)
* `base_outpath`: the path to the root of baseline generation for FID, KID.
* `eval_json`: the path to a formatted json file for evaluation metadata

**Evaluation Stage**

```
python evaluate.py --gpu 0,1 --root {} --filter {} --concept_type {} --caption_target {}  --outpkl {} --base_outpath {} --eval_json {} --eval_stage
```

the same script with additional parameters: `--eval_stage`

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

```
python sample.py --ckpt {} --prompt "New Orleans House Galaxy Case" --ddim_steps 50 --outdir samples_eval --n_copies 200 
python src/filter.py --folder {} --impath ../assets/mem_images/orleans.png --outpath {}
```
where `folder` is the path to saved images, i.e., `{ckpt-path}/samples_eval/` and outpath is the folder to save the images which are different than the memorized image.