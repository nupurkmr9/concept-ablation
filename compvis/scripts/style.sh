python train.py -t --gpus 0,1 --concept_type style --caption_target "van gogh" --prompts assets/finetune_prompts/painting.txt --name "van_gogh"

python train.py -t --gpus 0,1 --concept_type style --caption_target "monet" --prompts assets/finetune_prompts/painting.txt --name "monet"

python train.py -t --gpus 0,1 --concept_type style --caption_target "greg rutkowski" --prompts assets/finetune_prompts/painting.txt --name "greg"

python train.py -t --gpus 0,1 --concept_type style --caption_target "salvador dali" --prompts assets/finetune_prompts/painting.txt --name "salvador"

