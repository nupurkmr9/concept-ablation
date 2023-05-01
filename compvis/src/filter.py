import argparse
import os
import shutil
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
)
small_288 = transforms.Compose([
    transforms.Resize(288),
    transforms.ToTensor(),
    normalize,
])
skew_320 = transforms.Compose([
    transforms.Resize([320, 320]),
    transforms.ToTensor(),
    normalize,
])


def isimage(path):
    if 'png' in path.lower() or 'jpg' in path.lower() or 'jpeg' in path.lower():
        return True


model = torch.jit.load(
    "../assets/pretrained_models/sscd_imagenet_mixup.torchscript.pt")


def filter(folder, outpath, unfiltered_path, impath, threshold=0.15,
           image_threshold=0.5, anchor_size=10, target_size=3, return_score=False):
    if (Path(folder) / 'images.txt').exists():
        with open(f'{folder}/images.txt', "r") as f:
            image_paths = f.read().splitlines()
        with open(f'{folder}/caption.txt', "r") as f:
            image_captions = f.read().splitlines()
    else:
        image_paths = [os.path.join(str(folder), file_path)
                       for file_path in os.listdir(folder) if isimage(file_path)]
        image_captions = ["None" for _ in range(len(image_paths))]

    batch = small_288(Image.open(impath).convert('RGB')).unsqueeze(0)
    embedding_target = model(batch)[0, :]

    filtered_paths = []
    filtered_captions = []
    unfiltered_paths = []
    unfiltered_captions = []
    count_dict = {}
    for im, c in zip(image_paths, image_captions):
        if c not in count_dict:
            count_dict[c] = 0

        batch = small_288(Image.open(im).convert('RGB')).unsqueeze(0)
        embedding = model(batch)[0, :]

        diff_sscd = (embedding * embedding_target).sum()

        if diff_sscd <= image_threshold:
            filtered_paths.append(im)
            filtered_captions.append(c)
            count_dict[c] += 1
        else:
            unfiltered_paths.append(im)
            unfiltered_captions.append(c)

    # only return score
    if return_score:
        score = len(unfiltered_paths) / \
            (len(unfiltered_paths)+len(filtered_paths))
        print(len(unfiltered_paths))
        print(len(unfiltered_paths)+len(filtered_paths))
        print(f"ratio: {score}")
        return score

    os.makedirs(outpath, exist_ok=True)
    os.makedirs(f'{outpath}/samples', exist_ok=True)
    with open(f'{outpath}/caption.txt', 'w') as f:
        for each in filtered_captions:
            f.write(each.strip() + '\n')

    with open(f'{outpath}/images.txt', 'w') as f:
        for each in filtered_paths:
            f.write(each.strip() + '\n')
            imbase = Path(each).name
            shutil.copy(each, f'{outpath}/samples/{imbase}')

    # os.makedirs(unfiltered_path, exist_ok=True)
    # with open(f'{unfiltered_path}/caption.txt', 'w') as f:
    #     for each in unfiltered_captions:
    #         f.write(each.strip() + '\n')
    # with open(f'{unfiltered_path}/images.txt', 'w') as f:
    #     for each in unfiltered_paths:
    #         f.write(each.strip() + '\n')
    #         imbase = Path(each).name
    #         shutil.copy(each, f'{unfiltered_path}/{imbase}')
    print('++++++++++++++++++++++++++++++++++++++++++++++++')
    print('+ Filter Summary +')
    print(f'+ Remained images: {len(filtered_paths)}')
    print(f'+ Filtered images: {len(unfiltered_paths)}')
    print('++++++++++++++++++++++++++++++++++++++++++++++++')

    sorted_list = sorted(list(count_dict.items()),
                         key=lambda x: x[1], reverse=True)
    anchor_prompts = [c[0] for c in sorted_list[:anchor_size]]
    target_prompts = [c[0] for c in sorted_list[-target_size:]]
    return anchor_prompts, target_prompts


def parse_args():
    parser = argparse.ArgumentParser('', add_help=False)
    parser.add_argument('--folder', help='path to input images',
                        type=str)
    parser.add_argument('--impath', help='path to input image that is memorized',
                        type=str)
    parser.add_argument('--outpath', help='path to save images which are not memorized', default='./',
                        type=str)
    parser.add_argument('--threshold', help='threshold above which to keep images', default=0.15,
                        type=float)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    filter(args.folder, args.outpath, args.outpath,
           args.impath, args.threshold, return_score=True)
