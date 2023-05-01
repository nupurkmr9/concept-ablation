python train.py -t --gpus 0,1 --concept_type memorization --caption_target "New Orleans House Galaxy Case" --prompts ../assets/finetune_prompts/orleans_mem.txt --name "orleans" --mem_impath ../assets/mem_images/orleans.png --regularization

python train.py -t --gpus 0,1 --concept_type memorization --caption_target "Portrait of Tiger in black and white by Lukas Holas" --prompts ../assets/finetune_prompts/lukas_mem.txt --name "lukas" --mem_impath ../assets/mem_images/lukas.png --regularization

python train.py -t --gpus 0,1 --concept_type memorization --caption_target "VAN GOGH CAFE TERASSE copy.jpg" --prompts ../assets/finetune_prompts/vangoghcafe_mem.txt --name "vangoghcafe" --mem_impath ../assets/mem_images/vangoghcafe.png --regularization

python train.py -t --gpus 0,1 --concept_type memorization --caption_target "Captain Marvel Exclusive Ccxp Poster Released Online By Marvel" --prompts ../assets/finetune_prompts/marvel_mem.txt --name "marvel" --mem_impath ../assets/mem_images/marvel.png --regularization

python train.py -t --gpus 0,1 --concept_type memorization --caption_target "Sony Boss Confirms Bloodborne Expansion is Coming" --prompts ../assets/finetune_prompts/bloodborne_mem.txt --name "bloodborne" --mem_impath ../assets/mem_images/bloodborne.png --regularization

python train.py -t --gpus 0,1 --concept_type memorization --caption_target "<i>The Long Dark</i> Gets First Trailer, Steam Early Access" --prompts ../assets/finetune_prompts/longdark_mem.txt --name "longdark" --mem_impath ../assets/mem_images/longdark.png --regularization

python train.py -t --gpus 0,1 --concept_type memorization --caption_target "Ann Graham Lotz" --prompts ../assets/finetune_prompts/anne_mem.txt --name "anne" --mem_impath ../assets/mem_images/anne.png --regularization --base_lr 5e-7 --train_max_steps 210

python train.py -t --gpus 0,1 --concept_type memorization --caption_target "A painting with letter M written on it Canvas Wall Art Print" --prompts ../assets/finetune_prompts/letterm_mem.txt --name "letterm" --mem_impath ../assets/mem_images/letterm.png --regularization

