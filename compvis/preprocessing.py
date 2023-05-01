from pathlib import Path

import numpy as np
import torch.multiprocessing as mp
from src import utils
from src.filter import filter
from src.utils import safe_dir


def preprocess(opt, prompts, outdir, concept_type, ranks):
    """
        distribute generate training images from given initial prompts
    """
    mp.set_start_method('spawn')
    if concept_type == 'memorization':
        # load initial prompts
        with open(prompts, "r") as f:
            data = f.read().splitlines()
            prompt_size = len(data)
            data = np.array([5 * [prompt] for prompt in data]
                            ).reshape(-1, opt.n_samples).tolist()
        # check generation
        check_dir = safe_dir(Path(outdir) / 'check')
        check_sample_path = safe_dir(check_dir / 'samples')
        if len(list(check_sample_path.glob('*'))) != prompt_size * 5:
            utils.distributed_sample_images(
                data, ranks, opt.base[0], opt.resume_from_checkpoint_custom,
                opt.delta_ckpt, str(check_dir), 100
            )
        # filter and generate anchor and target prompts
        anchor_file_path = Path(outdir) / 'anchor.txt'
        target_file_path = Path(outdir) / 'target.txt'
        if not anchor_file_path.exists() or not target_file_path.exists():
            filter_dir = safe_dir(Path(outdir) / 'filtered')
            unfilter_dir = safe_dir(Path(outdir) / 'unfiltered')
            anchor_prompts, target_prompts = filter(str(check_dir), str(filter_dir),
                                                    str(unfilter_dir), opt.mem_impath)
            with open(anchor_file_path, 'w') as f:
                for prompt in anchor_prompts:
                    f.write(prompt.strip() + '\n')
            with open(target_file_path, 'w') as f:
                f.write(opt.caption_target + '\n')
                for prompt in target_prompts:
                    f.write(prompt.strip() + '\n')
        else:
            with open(anchor_file_path, 'r') as f:
                anchor_prompts = f.read().splitlines()
            with open(target_file_path, 'r') as f:
                target_prompts = f.read().splitlines()
        print(anchor_prompts)

        # generate new images
        assert opt.train_size % len(anchor_prompts) == 0
        n_repeat = opt.train_size // len(anchor_prompts)
        anchor_data = np.array(
            [n_repeat * [prompt] for prompt in anchor_prompts]).reshape(-1, opt.n_samples).tolist()
        anchor_dir = safe_dir(Path(outdir) / 'anchor')
        anchor_sample_path = safe_dir(anchor_dir / 'samples')
        if len(list(anchor_sample_path.glob('*'))) != opt.train_size:
            utils.distributed_sample_images(
                anchor_data, ranks, opt.base[0], opt.resume_from_checkpoint_custom,
                opt.delta_ckpt, str(anchor_dir), 200
            )
        # final filtering
        unfiltered_anchor_dir = safe_dir(Path(outdir) / 'un_filtered_anchor')
        if len(list(unfiltered_anchor_dir.glob('*'))) == 0:
            print('Final Filtering!')
            filter(str(anchor_dir), str(outdir), str(
                unfiltered_anchor_dir), opt.mem_impath)

        # update "target caption"
        if opt.caption_target is not None:
            caption_target = '*+'+opt.caption_target
            for c in target_prompts:
                caption_target += ';*+' + c
            opt.caption_target = caption_target
            print(opt.caption_target)
            print('++++++++++++++++++++')

    else:
        with open(opt.prompts, "r") as f:
            data = f.read().splitlines()
            assert opt.train_size % len(data) == 0
            n_repeat = opt.train_size // len(data)
            data = np.array([n_repeat * [prompt] for prompt in data]
                            ).reshape(-1, opt.n_samples).tolist()
        # check integrity
        sample_path = safe_dir(outdir / 'samples')
        if not sample_path.exists() or not len(list(sample_path.glob('*'))) == opt.train_size:
            utils.distributed_sample_images(
                data, ranks, opt.base[0], opt.resume_from_checkpoint_custom,
                opt.delta_ckpt, str(outdir), 200
            )
