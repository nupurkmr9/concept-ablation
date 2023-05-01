import glob
import hashlib
import os
import random
import shutil
from io import BytesIO
from pathlib import Path

import numpy as np
import openai
import regex as re
import requests
import torch
from clip_retrieval.clip_client import ClipClient
from diffusers import DPMSolverMultistepScheduler
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
)
small_288 = transforms.Compose([
    transforms.Resize(288),
    transforms.ToTensor(),
    normalize,
])


def retrieve(class_prompt, class_images_dir, num_class_images):
    factor = 1.5
    num_images = int(factor * num_class_images)
    client = ClipClient(url="https://knn.laion.ai/knn-service",
                        indice_name="laion_400m", num_images=num_images, aesthetic_weight=0.1)

    os.makedirs(f'{class_images_dir}/images', exist_ok=True)
    if len(list(Path(f'{class_images_dir}/images').iterdir())) >= num_class_images:
        return

    while True:
        class_images = client.query(text=class_prompt)
        if len(class_images) >= factor*num_class_images or num_images > 1e4:
            break
        else:
            num_images = int(factor * num_images)
            client = ClipClient(url="https://knn.laion.ai/knn-service",
                                indice_name="laion_400m", num_images=num_images, aesthetic_weight=0.1)

    count = 0
    total = 0
    pbar = tqdm(desc='downloading real regularization images',
                total=num_class_images)

    with open(f'{class_images_dir}/caption.txt', 'w') as f1, open(f'{class_images_dir}/urls.txt', 'w') as f2, open(f'{class_images_dir}/images.txt', 'w') as f3:
        while total < num_class_images:
            images = class_images[count]
            count += 1
            try:
                img = requests.get(images['url'])
                if img.status_code == 200:
                    _ = Image.open(BytesIO(img.content))
                    with open(f'{class_images_dir}/images/{total}.jpg', 'wb') as f:
                        f.write(img.content)
                    f1.write(images['caption'] + '\n')
                    f2.write(images['url'] + '\n')
                    f3.write(f'{class_images_dir}/images/{total}.jpg' + '\n')
                    total += 1
                    pbar.update(1)
                else:
                    continue
            except:
                continue
    return


def collate_fn(examples, with_prior_preservation):
    input_ids = [example["instance_prompt_ids"] for example in examples]
    input_anchor_ids = [example["instance_anchor_prompt_ids"]
                        for example in examples]
    pixel_values = [example["instance_images"] for example in examples]
    mask = [example["mask"] for example in examples]
    # Concat class and instance examples for prior preservation.
    # We do this to avoid doing two forward passes.
    if with_prior_preservation:
        input_ids += [example["class_prompt_ids"] for example in examples]
        pixel_values += [example["class_images"] for example in examples]
        mask += [example["class_mask"] for example in examples]

    input_ids = torch.cat(input_ids, dim=0)
    input_anchor_ids = torch.cat(input_anchor_ids, dim=0)
    pixel_values = torch.stack(pixel_values)
    mask = torch.stack(mask)
    pixel_values = pixel_values.to(
        memory_format=torch.contiguous_format).float()
    mask = mask.to(memory_format=torch.contiguous_format).float()

    batch = {
        "input_ids": input_ids,
        "input_anchor_ids": input_anchor_ids,
        "pixel_values": pixel_values,
        "mask": mask.unsqueeze(1)
    }
    return batch


class PromptDataset(Dataset):
    "A simple dataset to prepare the prompts to generate class images on multiple GPUs."

    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt[index % len(self.prompt)]
        example["index"] = index
        return example


class CustomDiffusionDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        concepts_list,
        concept_type,
        tokenizer,
        size=512,
        center_crop=False,
        with_prior_preservation=False,
        num_class_images=200,
        hflip=False,
        aug=True,
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer
        self.interpolation = Image.BILINEAR
        self.aug = aug
        self.concept_type = concept_type

        self.instance_images_path = []
        self.class_images_path = []
        self.with_prior_preservation = with_prior_preservation
        for concept in concepts_list:
            with open(concept["instance_data_dir"], "r") as f:
                inst_images_path = f.read().splitlines()
            with open(concept["instance_prompt"], "r") as f:
                inst_prompt = f.read().splitlines()
            inst_img_path = [(x, y, concept['caption_target'])
                             for (x, y) in zip(inst_images_path, inst_prompt)]
            self.instance_images_path.extend(inst_img_path)

            if with_prior_preservation:
                class_data_root = Path(concept["class_data_dir"])
                if os.path.isdir(class_data_root):
                    class_images_path = list(class_data_root.iterdir())
                    class_prompt = [concept["class_prompt"]
                                    for _ in range(len(class_images_path))]
                else:
                    with open(class_data_root, "r") as f:
                        class_images_path = f.read().splitlines()
                    with open(concept["class_prompt"], "r") as f:
                        class_prompt = f.read().splitlines()

                class_img_path = [(x, y) for (x, y) in zip(
                    class_images_path, class_prompt)]
                self.class_images_path.extend(
                    class_img_path[:num_class_images])

        random.shuffle(self.instance_images_path)
        self.num_instance_images = len(self.instance_images_path)
        self.num_class_images = len(self.class_images_path)
        self._length = max(self.num_class_images, self.num_instance_images)
        self.flip = transforms.RandomHorizontalFlip(0.5 * hflip)

        self.image_transforms = transforms.Compose(
            [
                self.flip,
                transforms.Resize(
                    size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(
                    size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def preprocess(self, image, scale, resample):
        outer, inner = self.size, scale
        if scale > self.size:
            outer, inner = scale, self.size
        top, left = np.random.randint(
            0, outer - inner + 1), np.random.randint(0, outer - inner + 1)
        image = image.resize((scale, scale), resample=resample)
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)
        instance_image = np.zeros((self.size, self.size, 3), dtype=np.float32)
        mask = np.zeros((self.size // 8, self.size // 8))
        if scale > self.size:
            instance_image = image[top: top + inner, left: left + inner, :]
            mask = np.ones((self.size // 8, self.size // 8))
        else:
            instance_image[top: top + inner, left: left + inner, :] = image
            mask[top // 8 + 1: (top + scale) // 8 - 1, left //
                 8 + 1: (left + scale) // 8 - 1] = 1.
        return instance_image, mask

    def __getprompt__(self, instance_prompt, instance_target):
        if self.concept_type == 'style':
            r = np.random.choice([0, 1, 2])
            instance_prompt = f'{instance_prompt}, in the style of {instance_target}' if r == 0 else f'in {instance_target}\'s style, {instance_prompt}' if r == 1 else f'in {instance_target}\'s style, {instance_prompt}'
        elif self.concept_type == 'object':
            anchor, target = instance_target.split('+')
            instance_prompt = instance_prompt.replace(anchor, target)
        elif self.concept_type == 'memorization':
            instance_prompt = instance_target.split('+')[1]
        return instance_prompt

    def __getitem__(self, index):
        example = {}
        instance_image, instance_prompt, instance_target = self.instance_images_path[
            index % self.num_instance_images]
        instance_image = Image.open(instance_image)
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        instance_image = self.flip(instance_image)
        # modify instance prompt according to the concept_type to include target concept
        # multiple style/object fine-tuning
        if ';' in instance_target:
            instance_target = instance_target.split(';')
            instance_target = instance_target[index % len(instance_target)]

        instance_anchor_prompt = instance_prompt
        instance_prompt = self.__getprompt__(instance_prompt, instance_target)
        # apply resize augmentation and create a valid image region mask
        random_scale = self.size
        if self.aug:
            random_scale = np.random.randint(self.size // 3, self.size + 1) if np.random.uniform(
            ) < 0.66 else np.random.randint(int(1.2 * self.size), int(1.4 * self.size))
        instance_image, mask = self.preprocess(
            instance_image, random_scale, self.interpolation)

        if random_scale < 0.6 * self.size:
            instance_prompt = np.random.choice(
                ["a far away ", "very small "]) + instance_prompt
        elif random_scale > self.size:
            instance_prompt = np.random.choice(
                ["zoomed in ", "close up "]) + instance_prompt

        example["instance_images"] = torch.from_numpy(
            instance_image).permute(2, 0, 1)
        example["mask"] = torch.from_numpy(mask)

        example["instance_prompt_ids"] = self.tokenizer(
            instance_prompt,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids
        example["instance_anchor_prompt_ids"] = self.tokenizer(
            instance_anchor_prompt,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids

        if self.with_prior_preservation:
            class_image, class_prompt = self.class_images_path[index %
                                                               self.num_class_images]
            class_image = Image.open(class_image)
            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")
            example["class_images"] = self.image_transforms(class_image)
            example["class_mask"] = torch.ones_like(example["mask"])
            example["class_prompt_ids"] = self.tokenizer(
                class_prompt,
                truncation=True,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids

        return example


def isimage(path):
    if 'png' in path.lower() or 'jpg' in path.lower() or 'jpeg' in path.lower():
        return True


def filter(folder, impath, outpath=None, unfiltered_path=None, threshold=0.15,
           image_threshold=0.5, anchor_size=10, target_size=3, return_score=False):
    model = torch.jit.load(
        "../assets/pretrained_models/sscd_imagenet_mixup.torchscript.pt")
    if isinstance(folder, list):
        image_paths = folder
        image_captions = ["None" for _ in range(len(image_paths))]
    elif Path(folder / 'images.txt').exists():
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
        if isinstance(folder, list):
            batch = small_288(im).unsqueeze(0)
        else:
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

    print('++++++++++++++++++++++++++++++++++++++++++++++++')
    print('+ Filter Summary +')
    print(f'+ Remained images: {len(filtered_paths)}')
    print(f'+ Filtered images: {len(unfiltered_paths)}')
    print('++++++++++++++++++++++++++++++++++++++++++++++++')

    sorted_list = sorted(list(count_dict.items()),
                         key=lambda x: x[1], reverse=True)
    anchor_prompts = [c[0] for c in sorted_list[:anchor_size]]
    target_prompts = [c[0] for c in sorted_list[-target_size:]]
    return anchor_prompts, target_prompts, len(filtered_paths)


def getanchorprompts(pipeline, accelerator, class_prompt, concept_type, class_images_dir, num_class_images=200, mem_impath=None):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    class_prompt_collection = []
    caption_target = []
    if concept_type == 'object':
        messages = [{"role": "system", "content": "You can describe any image via text and provide captions for wide variety of images that is possible to generate."}]
        messages = [{"role": "user", "content": f"Generate {num_class_images} captions for images containing a {class_prompt}. The caption should also contain the word \"{class_prompt}\" "}]
        while True:
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages
            )
            class_prompt_collection += [x for x in completion.choices[0].message.content.lower(
            ).split('\n') if class_prompt in x]
            messages.append(
                {"role": "assistant", "content": completion.choices[0].message.content})
            messages.append(
                {"role": "user", "content": f"Generate {num_class_images-len(class_prompt_collection)} more captions"})
            if len(class_prompt_collection) >= num_class_images:
                break
        class_prompt_collection = clean_prompt(class_prompt_collection)[
            :num_class_images]

    elif concept_type == 'memorization':
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
            pipeline.scheduler.config)
        num_prompts_firstpass = 5
        num_prompts_secondpass = 2
        threshold = 0.3
        # Generate num_prompts_firstpass paraphrases which generate different content at least 1-threshold % of the times.
        os.makedirs(class_images_dir / 'temp/', exist_ok=True)
        class_prompt_collection_counter = []
        caption_target = []
        prev_captions = []
        messages = [{"role": "user", "content": f"Generate {4*num_prompts_firstpass} different paraphrase of the caption: {class_prompt}. Preserve the meaning when paraphrasing."}]
        while True:
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages
            )
            print(completion.choices[0].message.content.lower().split('\n'))
            class_prompt_collection_ = [x.strip(
            ) for x in completion.choices[0].message.content.lower().split('\n') if x.strip() != '']
            class_prompt_collection_ = clean_prompt(class_prompt_collection_)
            print(class_prompt_collection_)
            for prompt in tqdm(
                class_prompt_collection_, desc="Generating anchor and target prompts ", disable=not accelerator.is_local_main_process
            ):
                print(f'Prompt: {prompt}')
                images = pipeline([prompt]*10, num_inference_steps=25,).images

                score = filter(images, mem_impath, return_score=True)
                print(f'Memorization rate: {score}')
                if score <= threshold and prompt not in class_prompt_collection and len(class_prompt_collection) < num_prompts_firstpass:
                    class_prompt_collection += [prompt]
                    class_prompt_collection_counter += [score]
                elif score >= 0.6 and prompt not in caption_target and len(caption_target) < 2:
                    caption_target += [prompt]
                if len(class_prompt_collection) >= num_prompts_firstpass and len(caption_target) >= 2:
                    break

            if len(class_prompt_collection) >= num_prompts_firstpass:
                break
            # print("prompts till now", class_prompt_collection, caption_target)
            # print("prompts till now", len(
            #     class_prompt_collection), len(caption_target))
            prev_captions += class_prompt_collection_
            prev_captions_ = ','.join(prev_captions[-40:])

            messages = [
                {"role": "user", "content": f"Generate {4*(num_prompts_firstpass- len(class_prompt_collection))} different paraphrase of the caption: {class_prompt}. Preserve the meaning the most when paraphrasing. Also make sure that the new captions are different from the following captions: {prev_captions_[:4000]}"}]

        # Generate more paraphrases using the captions we retrieved above.
        for prompt in class_prompt_collection[:num_prompts_firstpass]:
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": f"Generate {num_prompts_secondpass} different paraphrases of: {prompt}. "}]

            )
            class_prompt_collection += clean_prompt(
                [x.strip() for x in completion.choices[0].message.content.lower().split('\n') if x.strip() != ''])

        for prompt in tqdm(class_prompt_collection[num_prompts_firstpass:], desc="Memorization rate for final prompts"):
            images = pipeline([prompt]*10, num_inference_steps=25,).images

            class_prompt_collection_counter += [
                filter(images, mem_impath, return_score=True)]

        # select least ten and most memorized text prompts to be selected as anchor and target prompts.
        class_prompt_collection = sorted(
            zip(class_prompt_collection, class_prompt_collection_counter), key=lambda x: x[1])
        caption_target += [x for (x, y) in class_prompt_collection if y >= 0.6]
        class_prompt_collection = [
            x for (x, y) in class_prompt_collection if y <= threshold][:10]
        print(class_prompt_collection, caption_target)
    return class_prompt_collection, ';*+'.join(caption_target)


def clean_prompt(class_prompt_collection):
    class_prompt_collection = [re.sub(
        r"[0-9]+", lambda num: '' * len(num.group(0)), prompt) for prompt in class_prompt_collection]
    class_prompt_collection = [re.sub(
        r"^\.+", lambda dots: '' * len(dots.group(0)), prompt) for prompt in class_prompt_collection]
    class_prompt_collection = [x.strip() for x in class_prompt_collection]
    return class_prompt_collection


def safe_dir(dir):
    if not dir.exists():
        dir.mkdir()
    return dir
