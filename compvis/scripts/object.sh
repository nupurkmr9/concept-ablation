python train.py -t --gpus 0,1 --concept_type object --caption_target "cat+grumpy cat" --prompts ../assets/finetune_prompts/cat.txt --name "grumpy_cat" --regularization 
 
python train.py -t --gpus 0,1 --concept_type object --caption_target "robot+r2d2" --prompts ../assets/finetune_prompts/robot.txt --name "r2d2"  

python train.py -t --gpus 0,1 --concept_type object --caption_target "fish+nemo" --prompts ../assets/finetune_prompts/fish.txt --name "nemo" 

python train.py -t --gpus 0,1 --concept_type object --caption_target "dog+snoopy" --prompts ../assets/finetune_prompts/dog.txt --name "snoopy" 

