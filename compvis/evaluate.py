import argparse
import json
import os
import pathlib
import warnings
from itertools import islice
from pathlib import Path

import clip
import numpy as np
import pandas as pd
import sklearn.preprocessing
import torch
import torch.multiprocessing as mp
from cleanfid import fid
from ldm.util import instantiate_from_config
from packaging import version
from PIL import Image
from src import utils
from src.utils import safe_dir
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor
from tqdm import tqdm


class CLIPCapDataset(torch.utils.data.Dataset):
    def __init__(self, data, append=False, prefix='A photo depicts'):
        self.data = data
        self.prefix = ''
        if append:
            self.prefix = prefix
            if self.prefix[-1] != ' ':
                self.prefix += ' '

    def __getitem__(self, idx):
        c_data = self.data[idx]
        c_data = clip.tokenize(self.prefix + c_data, truncate=True).squeeze()
        return {'caption': c_data}

    def __len__(self):
        return len(self.data)


def Convert(image):
    return image.convert("RGB")


class CLIPImageDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
        # only 224x224 ViT-B/32 supported for now
        self.preprocess = self._transform_test(224)

    def _transform_test(self, n_px):
        return Compose([
            Resize(n_px, interpolation=Image.BICUBIC),
            CenterCrop(n_px),
            Convert,
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073),
                      (0.26862954, 0.26130258, 0.27577711)),
        ])

    def __getitem__(self, idx):
        c_data = self.data[idx]
        image = Image.open(c_data)
        image = self.preprocess(image)
        return {'image': image}

    def __len__(self):
        return len(self.data)


def extract_all_captions(captions, model, device, batch_size=256, num_workers=8, append=False):
    data = torch.utils.data.DataLoader(
        CLIPCapDataset(captions, append=append),
        batch_size=batch_size, num_workers=num_workers, shuffle=False)
    all_text_features = []
    with torch.no_grad():
        for b in tqdm(data):
            b = b['caption'].to(device)
            all_text_features.append(model.encode_text(b).cpu().numpy())
    all_text_features = np.vstack(all_text_features)
    return all_text_features


def extract_all_images(images, model, device, batch_size=64, num_workers=8):
    data = torch.utils.data.DataLoader(
        CLIPImageDataset(images),
        batch_size=batch_size, num_workers=num_workers, shuffle=False)
    all_image_features = []
    with torch.no_grad():
        for b in tqdm(data):
            b = b['image'].to(device)
            if device == 'cuda':
                b = b.to(torch.float16)
            all_image_features.append(model.encode_image(b).cpu().numpy())
    all_image_features = np.vstack(all_image_features)
    return all_image_features


def get_clip_score(model, images, candidates, device, append=False, w=2.5):
    '''
    get standard image-text clipscore.
    images can either be:
    - a list of strings specifying filepaths for images
    - a precomputed, ordered matrix of image features
    '''
    if isinstance(images, list):
        # need to extract image features
        images = extract_all_images(images, model, device)

    candidates = extract_all_captions(candidates, model, device, append=append)

    # as of numpy 1.21, normalize doesn't work properly for float16
    if version.parse(np.__version__) < version.parse('1.21'):
        images = sklearn.preprocessing.normalize(images, axis=1)
        candidates = sklearn.preprocessing.normalize(candidates, axis=1)
    else:
        warnings.warn(
            'due to a numerical instability, new numpy normalization is slightly different than paper results. '
            'to exactly replicate paper results, please use numpy version less than 1.21, e.g., 1.20.3.')
        images = images / np.sqrt(np.sum(images ** 2, axis=1, keepdims=True))
        candidates = candidates / \
            np.sqrt(np.sum(candidates ** 2, axis=1, keepdims=True))

    per = w * np.clip(np.sum(images * candidates, axis=1), 0, None)
    return np.mean(per), per, candidates


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)

    token_weights = sd["cond_stage_model.transformer.text_model.embeddings.token_embedding.weight"]
    del sd["cond_stage_model.transformer.text_model.embeddings.token_embedding.weight"]
    m, u = model.load_state_dict(sd, strict=False)
    model.cond_stage_model.transformer.text_model.embeddings.token_embedding.weight.data[
        :token_weights.shape[0]] = token_weights
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.eval()
    return model


# ----------------------------------------------------------------------------

def clipeval(image_dir, candidates_json, target_prompts, device):
    image_paths = [os.path.join(image_dir, path) for path in os.listdir(image_dir)
                   if path.endswith(('.png', '.jpg', '.jpeg', '.tiff'))]
    image_ids = [Path(path).stem for path in image_paths]
    model, transform = clip.load("ViT-B/32", device=device, jit=False)
    model.eval()

    image_feats = extract_all_images(
        image_paths, model, device, batch_size=64, num_workers=8)

    clipscores = []
    clipscores_std = []
    clipscores_all = []
    for i, each_prompt in enumerate(target_prompts):
        candidates = [f'{each_prompt}' for _ in image_ids]
        append = True
        if target_prompts[1] == 'painting':
            if each_prompt != 'painting':
                candidates = [f'{x} style' for x in candidates]

        _, per_instance_image_text, candidate_feats = get_clip_score(
            model, image_feats, candidates, device, append=append)
        scores_each_prompt = {image_id: {'CLIPScore': float(clipscore)}
                              for image_id, clipscore in
                              zip(image_ids, per_instance_image_text)}

        clipscores.append(np.mean([s['CLIPScore']
                          for s in scores_each_prompt.values()]))
        clipscores_std.append(np.std([s['CLIPScore']
                              for s in scores_each_prompt.values()]))
        clipscores_all.append(
            np.array([s['CLIPScore'] for s in scores_each_prompt.values()]))

    clipaccuracy = np.mean(clipscores_all[0] > clipscores_all[1])

    print("std:", clipscores_std)
    return clipscores[0], clipaccuracy


def clipeval_image(image_dir, image_dir_ref, device):
    with open(image_dir_ref, "r") as f:
        image_paths_ref = f.read().splitlines()

    print('+++++++++++++++')
    print("Reference images:", len(image_paths_ref))
    print('+++++++++++++++')
    size = min(len(list(pathlib.Path(image_dir).glob('*'))),
               5 * len(image_paths_ref))

    image_paths = [os.path.join(image_dir, f'{i:05}.png') for i in range(size)]

    model, transform = clip.load("ViT-B/32", device=device, jit=False)
    model.eval()

    image_feats = extract_all_images(
        image_paths, model, device, batch_size=64, num_workers=8)

    image_feats_ref = extract_all_images(
        image_paths_ref, model, device, batch_size=64, num_workers=8)

    image_feats = image_feats / \
        np.sqrt(np.sum(image_feats ** 2, axis=1, keepdims=True))
    image_feats_ref = image_feats_ref / \
        np.sqrt(np.sum(image_feats_ref ** 2, axis=1, keepdims=True))
    res = image_feats @ image_feats_ref.T
    return np.mean(res), None, None


def load_data(name, numgen, general=None, target=None, n_samples=10):
    file_path = '../assets/eval_prompts/{}_eval.txt'.format(name)
    with open(file_path, 'r') as f:
        data = f.read().splitlines()
        if general is not None and target is not None:
            data = [prompt.strip().replace(general, target) for prompt in data]
        n_repeat = numgen // len(data)
        data = np.array([n_repeat * [prompt]
                        for prompt in data]).reshape(-1, n_samples).tolist()
    return data


def prompt_json(path):
    prompts = {}
    with open(str(path/'caption.txt'), 'r') as f:
        data = f.read().splitlines()
    for i, prompt in enumerate(data):
        prompts[f"{i:05}"] = prompt
    json.dump(prompts, open(str(path/'prompts.json'), 'w'), indent=2)


def check_generation(folder, numgen):
    if not folder.exists():
        return False
    if not (folder/'samples').exists():
        return False
    return len(list((folder/'samples').glob('*'))) == numgen


def retrieve_target_prompts(type, target, eval_json):
    meta_data = json.load(open(eval_json, 'r'))[type][target]
    target_prompts = [meta_data['target'],
                      meta_data['anchor']] + meta_data['hard_negative']
    return [x.lower() for x in target_prompts]


def getmetrics(type, target, base_sample_root, ranks, ckpt, n_samples, config,
               sample_root, eval_json, numgen, base_ckpt="sd-v1-4.ckpt"):
    # load data
    meta_data = json.load(open(eval_json, 'r'))[type][target]
    target_prompts = retrieve_target_prompts(type, target, eval_json)
    numgens = 2 * [numgen * 4] + (len(target_prompts) - 2) * [numgen]
    for cur_target, cur_numgen in zip(target_prompts, numgens):
        # load data
        if type == 'style':
            data = load_data(cur_target.replace(
                ' ', '_').replace('-', '_'), cur_numgen, n_samples=n_samples)
        elif type == 'object':
            data = load_data(meta_data['anchor'].replace(
                ' ', '_').replace('-', '_'), cur_numgen,
                meta_data['anchor'], cur_target, n_samples=n_samples)
        else:
            raise NotImplementedError

        # model generation
        sample_path = safe_dir(
            sample_root / (ckpt.stem + "-" + cur_target.replace(' ', '_').replace('-', '_')))
        if not check_generation(sample_path, cur_numgen):
            utils.distributed_sample_images(
                data, ranks, config, str(
                    ckpt) if 'delta' not in str(ckpt) else base_ckpt,
                None if 'delta' not in str(ckpt) else str(ckpt), str(sample_path), ddim_steps=50)
            prompt_json(sample_path)

        # load data for baseline generation
        if type == 'style':
            data = load_data(cur_target.replace(
                ' ', '_').replace('-', '_'), numgen)
        elif type == 'object':
            data = load_data(meta_data['anchor'].replace(
                ' ', '_').replace('-', '_'), numgen,
                meta_data['anchor'], cur_target)
        else:
            raise NotImplementedError

        # base generation
        base_sample_path = safe_dir(Path(
            base_sample_root) / ("base-" + cur_target.replace(' ', '_').replace('-', '_')))
        if not check_generation(base_sample_path, numgen):
            utils.distributed_sample_images(
                data, ranks, config, base_ckpt,
                None, str(base_sample_path), ddim_steps=50)
            prompt_json(sample_path)


def calmetrics(target_prompts, sample_root, outpath, base_sample_root, ckpt):
    device = 'cuda'
    if os.path.exists(outpath):
        df = pd.read_pickle(outpath)
    else:
        df = pd.DataFrame()
    full = {}
    for cur_target in target_prompts:
        name = cur_target.replace(' ', '_').replace('-', '_')
        folder = sample_root / f'{ckpt.stem}-{name}'
        assert folder.is_dir()
        image_path = folder / 'samples'
        json_path = folder / 'prompts.json'
        concept_name = folder.name.split('-')[-1]
        # add the category name corresponding to the folder and the anchor category.
        prompts = [cur_target, target_prompts[1]]

        clipscores, clipaccuracy = \
            clipeval(str(image_path), str(json_path), prompts, device)
        data_location = '{}/base-{}/samples'.format(
            base_sample_root, concept_name)
        kidscore = fid.compute_kid(str(image_path), data_location)
        clipscores_baseline, clipaccuracy_baseline = \
            clipeval(str(data_location), None, prompts, device)
        sd = {}
        sd[f'KID_{prompts[0]}'] = [kidscore]
        sd[f'CLIP scores_{prompts[0]}_{prompts[0]}'] = clipscores
        sd[f'CLIP accuracy_{prompts[0]}_{prompts[1]}'] = clipaccuracy
        sd[f'Baseline CLIP scores_{prompts[0]}_{prompts[0]}'] = clipscores_baseline
        sd[f'Baseline CLIP accuracy_{prompts[0]}_{prompts[1]}'] = clipaccuracy_baseline
        expname = sample_root.parent.name + "_" + folder.name.split('-')[0]
        if expname not in full:
            full[expname] = sd
        else:
            full[expname] = {**sd, **full[expname]}

    print("Metrics:", full)

    for expname, sd in full.items():
        if expname not in df.index:
            df1 = pd.DataFrame(sd, index=[expname])
            df = pd.concat([df, df1])
        else:
            df.loc[df.index == expname, sd.keys()] = sd.values()

    df.to_pickle(outpath)


def parse_args():
    parser = argparse.ArgumentParser("metric", add_help=False)
    parser.add_argument("--root", type=str,
                        help="the root folder to trained model")
    parser.add_argument("--filter", type=str, default='step_*.ckpt',
                        help="the regular expression for models")
    parser.add_argument("--eval_path", type=str, default='eval',
                        help="the path to root of all generated images")
    parser.add_argument("--concept_type", type=str,
                        required=True, help="type of concept removed")
    parser.add_argument("--caption_target", type=str,
                        required=True, help="the target for ablated concept")
    parser.add_argument("--eval_json", type=str, default='../assets/eval.json',
                        help="the json file that stores metadata for evaluation")
    parser.add_argument("--numgen", type=int, default=50,
                        help="number of images for each hard negative (x4 for target and anchor concept).")
    parser.add_argument("--n_samples", type=int, default=10,
                        help="batch-size")
    parser.add_argument("--gpus", type=str, default='0,',
                        help="number of gpus")
    parser.add_argument("--config", type=str, default="configs/finetune.yaml",
                        help="path to config which constructs model")
    parser.add_argument("--outpkl", type=str, default="metrics/evaluation.pkl",
                        help="the path to save result pkl file")
    parser.add_argument("--base_ckpt", type=str, default="../assets/pretrained_models/sd-v1-4.ckpt",
                        help="the baseline model to compute fid and kid")
    parser.add_argument("--base_outpath", type=str, default="../assets/baseline_generation",
                        help="the path to saved generated baseline images")
    parser.add_argument("--eval_stage", action="store_true",
                        help="False: generation stage, True: evaluation stage")
    return parser.parse_args()


def main(args):
    sample_root = safe_dir(Path(args.root) / args.eval_path)
    os.makedirs(args.base_outpath, exist_ok=True)
    ranks = [int(i) for i in args.gpus.split(',') if i != ""]
    if not args.eval_stage:
        for ckpt in (Path(args.root) / 'checkpoints').glob(args.filter):
            print(ckpt)
            getmetrics(args.concept_type, args.caption_target, args.base_outpath, ranks, ckpt,
                       args.n_samples, args.config, sample_root, args.eval_json, args.numgen, args.base_ckpt)

    else:
        target_prompts = retrieve_target_prompts(
            args.concept_type, args.caption_target, args.eval_json
        )
        for ckpt in (Path(args.root) / 'checkpoints').glob(args.filter):
            calmetrics(target_prompts, sample_root,
                       args.outpkl, args.base_outpath, ckpt)


if __name__ == "__main__":
    # distributed setting
    args = parse_args()
    if not args.eval_stage:
        mp.set_start_method('spawn')
    main(args)
