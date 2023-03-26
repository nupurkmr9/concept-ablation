import argparse
import os
import sys
import glob
import pathlib
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
import time
from pathlib import Path
import json
from src.utils import safe_dir
import sklearn.preprocessing
import warnings
from packaging import version
from io import BytesIO
import torch.nn as nn
import json
import torch
import torch.multiprocessing as mp
from einops import rearrange
from torchvision.utils import make_grid
from torch import autocast
from pytorch_lightning import seed_everything
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import clip
from cleanfid import fid
import torch.nn.functional as F

from contextlib import contextmanager, nullcontext
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
import pandas as pd
from src import utils


class CLIPCapDataset(torch.utils.data.Dataset):
    def __init__(self, data, prefix='A photo depicts'):
        self.data = data
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
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    def __getitem__(self, idx):
        c_data = self.data[idx]
        image = Image.open(c_data)
        image = self.preprocess(image)
        return {'image': image}

    def __len__(self):
        return len(self.data)


class VGGImageDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
        # only 224x224 ViT-B/32 supported for now
        self.preprocess = self._transform_test(224)

    def _transform_test(self, n_px):
        return Compose([
            Resize(n_px, interpolation=Image.BICUBIC),
            CenterCrop(n_px),
            lambda image: image.convert("RGB"),
            ToTensor(),
            Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def __getitem__(self, idx):
        c_data = self.data[idx]
        image = Image.open(c_data)
        image = self.preprocess(image)
        return {'image': image}

    def __len__(self):
        return len(self.data)


def extract_all_captions(captions, model, device, batch_size=256, num_workers=8):
    data = torch.utils.data.DataLoader(
        CLIPCapDataset(captions),
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


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg19', pretrained=True)
        model.eval()
        enc_layers = list(model.features.children())
        self.enc_1 = nn.Sequential(*enc_layers[:4])  # input -> relu1_1
        self.enc_2 = nn.Sequential(*enc_layers[4:11])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*enc_layers[11:18])  # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*enc_layers[18:31])  # relu3_1 -> relu4_1
        # fix the encoder
        for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False

    # extract relu1_1, relu2_1, relu3_1, relu4_1 from input image
    def encode_with_intermediate(self, input):
        results = [input]
        for i in range(4):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]


def extract_all_vgg_images(images, device, batch_size=64, num_workers=8):
    model = Net()
    model = model.to(device)
    data = torch.utils.data.DataLoader(
        VGGImageDataset(images),
        batch_size=batch_size, num_workers=num_workers, shuffle=False)
    all_image_features = [[], [], [], []]
    with torch.no_grad():
        for b in tqdm(data):
            b = b['image'].to(device)
            if device == 'cuda':
                b = b.to(torch.float)
            feats = model.encode_with_intermediate(b)
            for i, feat in enumerate(feats):
                # print(feat.size())
                all_image_features[i].append(feat.cpu())
    all_image_features = [torch.cat(x, 0) for x in all_image_features]
    return all_image_features


def get_clip_score(model, images, candidates, device, w=2.5):
    '''
    get standard image-text clipscore.
    images can either be:
    - a list of strings specifying filepaths for images
    - a precomputed, ordered matrix of image features
    '''
    if isinstance(images, list):
        # need to extract image features
        images = extract_all_images(images, model, device)

    candidates = extract_all_captions(candidates, model, device)

    # as of numpy 1.21, normalize doesn't work properly for float16
    if version.parse(np.__version__) < version.parse('1.21'):
        images = sklearn.preprocessing.normalize(images, axis=1)
        candidates = sklearn.preprocessing.normalize(candidates, axis=1)
    else:
        warnings.warn(
            'due to a numerical instability, new numpy normalization is slightly different than paper results. '
            'to exactly replicate paper results, please use numpy version less than 1.21, e.g., 1.20.3.')
        images = images / np.sqrt(np.sum(images ** 2, axis=1, keepdims=True))
        candidates = candidates / np.sqrt(np.sum(candidates ** 2, axis=1, keepdims=True))

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
    with open(candidates_json) as f:
        candidates = json.load(f)
    candidates = [candidates[cid] for cid in image_ids]
    # print(image_ids, candidates)

    model, transform = clip.load("ViT-B/32", device=device, jit=False)
    model.eval()

    image_feats = extract_all_images(
        image_paths, model, device, batch_size=64, num_workers=8)

    # get image-text clipscore
    _, per_instance_image_text, candidate_feats = get_clip_score(
        model, image_feats, candidates, device)

    scores = {image_id: {'CLIPScore': float(clipscore)}
              for image_id, clipscore in
              zip(image_ids, per_instance_image_text)}
    # print('CLIPScore: {:.4f}'.format(np.mean([s['CLIPScore'] for s in scores.values()])))

    clipscores = []
    clipscores_std = []
    clipscores_all = []
    for each_prompt in target_prompts:
        candidates = [f'{each_prompt}' for _ in image_ids]
        _, per_instance_image_text, candidate_feats = get_clip_score(
            model, image_feats, candidates, device)
        scores_each_prompt = {image_id: {'CLIPScore': float(clipscore)}
                              for image_id, clipscore in
                              zip(image_ids, per_instance_image_text)}

        clipscores.append(np.mean([s['CLIPScore'] for s in scores_each_prompt.values()]))
        clipscores_std.append(np.std([s['CLIPScore'] for s in scores_each_prompt.values()]))
        clipscores_all.append(np.array([s['CLIPScore'] for s in scores_each_prompt.values()]))

    clipaccuracy = []
    for i in range(len(clipscores_all) - 1):
        clipaccuracy.append(np.mean(clipscores_all[i] > clipscores_all[i - 1]))

    print("std:", clipscores_std)
    return np.mean([s['CLIPScore'] for s in scores.values()]), clipscores, clipaccuracy


def calc_mean_std_gram(feat, calc_gram=False, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    H, W = size[2:]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    feat_gram = None
    if calc_gram:
        feat_gram = torch.einsum('bnc,bmc->bnm', feat.view(N, C, -1).permute(0, 2, 1),
                                 feat.view(N, C, -1).permute(0, 2, 1)) / (H * W)
    return feat_mean, feat_std, feat_gram


def calc_style_loss(inputs, targets):
    loss = 0.
    loss_gram = 0.
    count = 0
    for (input, target) in zip(inputs, targets):
        input_mean, input_std, input_gram = calc_mean_std_gram(input, calc_gram=count > 0)
        target_mean, target_std, target_gram = calc_mean_std_gram(target, calc_gram=count > 0)
        loss1 = 0
        loss1_gram = 0.
        for i in range(10):
            loss1 += F.mse_loss(input_mean[i::10], target_mean) + F.mse_loss(input_std[i::10], target_std)
            if count > 0:
                loss1_gram += F.mse_loss(input_gram[i::10], target_gram)
        loss += loss1 / 10.
        loss_gram += loss1_gram / 10.
        count += 1
    return loss, loss_gram


def clipeval_image(image_dir, image_dir_ref, device):
    with open(image_dir_ref, "r") as f:
        image_paths_ref = f.read().splitlines()

    print('+++++++++++++++')
    print(len(image_paths_ref))
    print('+++++++++++++++')
    print(image_paths_ref)
    size = min(len(list(pathlib.Path(image_dir).glob('*'))), 5 * len(image_paths_ref))

    image_paths = [os.path.join(image_dir, f'{i:05}.png') for i in range(size)]

    model, transform = clip.load("ViT-B/32", device=device, jit=False)
    model.eval()

    image_feats = extract_all_images(
        image_paths, model, device, batch_size=64, num_workers=8)

    image_feats_ref = extract_all_images(
        image_paths_ref, model, device, batch_size=64, num_workers=8)

    image_feats = image_feats / np.sqrt(np.sum(image_feats ** 2, axis=1, keepdims=True))
    image_feats_ref = image_feats_ref / np.sqrt(np.sum(image_feats_ref ** 2, axis=1, keepdims=True))
    res = image_feats @ image_feats_ref.T
    return np.mean(res), None, None


def load_data(name, numgen, general=None, target=None, n_samples=10):
    file_path = 'assets/eval_prompts/{}_eval.txt'.format(name)
    with open(file_path, 'r') as f:
        data = f.read().splitlines()
        if general is not None and target is not None:
            data = [prompt.strip().replace(general, target) for prompt in data]
        n_repeat = numgen // len(data)
        data = np.array([n_repeat * [prompt] for prompt in data]).reshape(-1, n_samples).tolist()
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
    return [meta_data['target'], meta_data['anchor']] + meta_data['hard_negative']


def getmetrics(type, target, base_sample_root, ranks, ckpt, config,
               sample_root, eval_json, numgen, base_ckpt="model-v1-4.ckpt"):
    # load data
    meta_data = json.load(open(eval_json, 'r'))[type][target]
    target_prompts = retrieve_target_prompts(type, target, eval_json)
    numgens = 2 * [numgen * 4] + (len(target_prompts) - 2) * [numgen]
    for cur_target, cur_numgen in zip(target_prompts, numgens):
        # load data
        if type == 'style':
            data = load_data(cur_target, cur_numgen)
        elif type == 'object':
            data = load_data(meta_data['anchor'], cur_numgen,
                             meta_data['anchor'], cur_target)
        else:
            raise NotImplementedError

        # model generation
        sample_path = safe_dir(sample_root / (ckpt.stem + "-" + cur_target.replace(' ', '_').replace('-', '_')))
        if not check_generation(sample_path, cur_numgen):
            utils.distributed_sample_images(
                data, ranks, config, str(ckpt),
                None, str(sample_path), ddim_steps=50)
            prompt_json(sample_path)

        # load data
        if type == 'style':
            base_data = load_data(cur_target, numgen)
        elif type == 'object':
            base_data = load_data(meta_data['anchor'], numgen,
                                  meta_data['anchor'], cur_target)
        else:
            raise NotImplementedError

        # base generation
        base_sample_path = safe_dir(Path(base_sample_root) / ("base-" + cur_target.replace(' ', '_').replace('-', '_')))
        if not check_generation(base_sample_path, numgen):
            utils.distributed_sample_images(
                base_data, ranks, config, base_ckpt,
                None, str(base_sample_path), ddim_steps=50)
            prompt_json(sample_path)


def calmetrics(target_prompts, sample_root, outpath, base_sample_root):
    device = 'cuda'
    if os.path.exists(outpath):
        df = pd.read_pickle(outpath)
    else:
        df = pd.DataFrame()
    full = {}
    for folder in sample_root.glob('*'):
        assert folder.is_dir()
        image_path = folder / 'samples'
        json_path = folder / 'prompts.json'
        concept_name = folder.name.split('-')[-1]
        clipscore, clipscores, clipaccuracy = \
            clipeval(str(image_path), str(json_path), target_prompts, device)
        data_location = '{}/base-{}/samples'.format(base_sample_root, concept_name)
        fidscore = fid.compute_fid(str(image_path), data_location)
        kidscore = fid.compute_kid(str(image_path), data_location)
        sd = {}
        extension = concept_name.replace('_', '')
        sd[f'FID_{extension}'] = [fidscore]
        sd[f'KID_{extension}'] = [kidscore]
        for (x, y) in zip(clipscores, target_prompts):
            sd[f'CLIP scores_{extension}_{y.replace(" ", "")}'] = [x]
        for (x, y) in zip(clipaccuracy, target_prompts):
            sd[f'CLIP accuracy_{extension}_{y.replace(" ", "")}'] = [x]
        expname = sample_root.parent.name + "_" + folder.name.split('-')[0]
        if expname not in full:
            full[expname] = sd
        else:
            full[expname] = {**sd, **full[expname]}

    for expname, sd in full.items():
        if expname not in df.index:
            df1 = pd.DataFrame(sd, index=[expname])
            df = pd.concat([df, df1])
        else:
            df.loc[df.index == expname, sd.keys()] = sd.values()
    df.to_pickle(outpath)


def parse_args():
    parser = argparse.ArgumentParser("metric", add_help=False)
    parser.add_argument("--root", type=str, help="the root folder to trained model")
    parser.add_argument("--filter", type=str, default='step_*.ckpt', help="the regular expression for models")
    parser.add_argument("--eval_path", type=str, default='eval', help="the path to root of all generated images")
    parser.add_argument("--concept_type", type=str, required=True, help="type of concept removed")
    parser.add_argument("--caption_target", type=str, required=True, help="the target for ablated concept")
    parser.add_argument("--eval_json", type=str, default='assets/eval.json',
                        help="the json file that stores metadata for evaluation")
    parser.add_argument("--numgen", type=int, default=50,
                        help="number of images for each hard negative (x4 for target and general).")
    parser.add_argument("--gpus", type=str, default='0,', help="number of gpus")
    parser.add_argument("--config", type=str, default="configs/finetune.yaml",
                        help="path to config which constructs model")
    parser.add_argument("--outpkl", type=str, default="metrics/evaluation.pkl",
                        help="the path to save result pkl file")
    parser.add_argument("--base_ckpt", type=str, default="assets/pretrained_models/model-v1-4.ckpt",
                        help="the baseline model to compute fid and kid")
    parser.add_argument("--base_outpath", type=str, default="assets/baseline_generation",
                        help="the path to saved generated baseline images")
    parser.add_argument("--eval_stage", action="store_true",
                        help="False: generation stage, True: evaluation stage")
    return parser.parse_args()


def main(args):
    sample_root = safe_dir(Path(args.root) / args.eval_path)
    ranks = [int(i) for i in args.gpus.split(',') if i != ""]
    if not args.eval_stage:
        for ckpt in (Path(args.root) / 'checkpoints').glob(args.filter):
            print(ckpt)
            getmetrics(args.concept_type, args.caption_target, args.base_outpath,
                       ranks, ckpt, args.config, sample_root, args.eval_json, args.numgen, args.base_ckpt)

    else:
        target_prompts = retrieve_target_prompts(
            args.concept_type, args.caption_target, args.eval_json
        )
        calmetrics(target_prompts, sample_root, args.outpkl, args.base_outpath)


if __name__ == "__main__":
    # distributed setting
    args = parse_args()
    if not args.eval_stage:
        mp.set_start_method('spawn')
    main(args)
