# This code is built from the Stable Diffusion repository: https://github.com/CompVis/stable-diffusion.
# Copyright (c) 2022 Robin Rombach and Patrick Esser and contributors.
# CreativeML Open RAIL-M
#
# ==========================================================================================
#
# Adobe’s modifications are Copyright 2022 Adobe Research. All rights reserved.
# Adobe’s modifications are licensed under the Adobe Research License. To view a copy of the license, visit
# LICENSE.
#
# ==========================================================================================
#
# CreativeML Open RAIL-M License
#
# Section I: PREAMBLE

# Multimodal generative models are being widely adopted and used, and have the potential to transform the way artists, among other individuals, conceive and benefit from AI or ML technologies as a tool for content creation.

# Notwithstanding the current and potential benefits that these artifacts can bring to society at large, there are also concerns about potential misuses of them, either due to their technical limitations or ethical considerations.

# In short, this license strives for both the open and responsible downstream use of the accompanying model. When it comes to the open character, we took inspiration from open source permissive licenses regarding the grant of IP rights. Referring to the downstream responsible use, we added use-based restrictions not permitting the use of the Model in very specific scenarios, in order for the licensor to be able to enforce the license in case potential misuses of the Model may occur. At the same time, we strive to promote open and responsible research on generative models for art and content generation.

# Even though downstream derivative versions of the model could be released under different licensing terms, the latter will always have to include - at minimum - the same use-based restrictions as the ones in the original license (this license). We believe in the intersection between open and responsible AI development; thus, this License aims to strike a balance between both in order to enable responsible open-science in the field of AI.

# This License governs the use of the model (and its derivatives) and is informed by the model card associated with the model.

# NOW THEREFORE, You and Licensor agree as follows:

# 1. Definitions

# - "License" means the terms and conditions for use, reproduction, and Distribution as defined in this document.
# - "Data" means a collection of information and/or content extracted from the dataset used with the Model, including to train, pretrain, or otherwise evaluate the Model. The Data is not licensed under this License.
# - "Output" means the results of operating a Model as embodied in informational content resulting therefrom.
# - "Model" means any accompanying machine-learning based assemblies (including checkpoints), consisting of learnt weights, parameters (including optimizer states), corresponding to the model architecture as embodied in the Complementary Material, that have been trained or tuned, in whole or in part on the Data, using the Complementary Material.
# - "Derivatives of the Model" means all modifications to the Model, works based on the Model, or any other model which is created or initialized by transfer of patterns of the weights, parameters, activations or output of the Model, to the other model, in order to cause the other model to perform similarly to the Model, including - but not limited to - distillation methods entailing the use of intermediate data representations or methods based on the generation of synthetic data by the Model for training the other model.
# - "Complementary Material" means the accompanying source code and scripts used to define, run, load, benchmark or evaluate the Model, and used to prepare data for training or evaluation, if any. This includes any accompanying documentation, tutorials, examples, etc, if any.
# - "Distribution" means any transmission, reproduction, publication or other sharing of the Model or Derivatives of the Model to a third party, including providing the Model as a hosted service made available by electronic or other remote means - e.g. API-based or web access.
# - "Licensor" means the copyright owner or entity authorized by the copyright owner that is granting the License, including the persons or entities that may have rights in the Model and/or distributing the Model.
# - "You" (or "Your") means an individual or Legal Entity exercising permissions granted by this License and/or making use of the Model for whichever purpose and in any field of use, including usage of the Model in an end-use application - e.g. chatbot, translator, image generator.
# - "Third Parties" means individuals or legal entities that are not under common control with Licensor or You.
# - "Contribution" means any work of authorship, including the original version of the Model and any modifications or additions to that Model or Derivatives of the Model thereof, that is intentionally submitted to Licensor for inclusion in the Model by the copyright owner or by an individual or Legal Entity authorized to submit on behalf of the copyright owner. For the purposes of this definition, "submitted" means any form of electronic, verbal, or written communication sent to the Licensor or its representatives, including but not limited to communication on electronic mailing lists, source code control systems, and issue tracking systems that are managed by, or on behalf of, the Licensor for the purpose of discussing and improving the Model, but excluding communication that is conspicuously marked or otherwise designated in writing by the copyright owner as "Not a Contribution."
# - "Contributor" means Licensor and any individual or Legal Entity on behalf of whom a Contribution has been received by Licensor and subsequently incorporated within the Model.

# Section II: INTELLECTUAL PROPERTY RIGHTS

# Both copyright and patent grants apply to the Model, Derivatives of the Model and Complementary Material. The Model and Derivatives of the Model are subject to additional terms as described in Section III.

# 2. Grant of Copyright License. Subject to the terms and conditions of this License, each Contributor hereby grants to You a perpetual, worldwide, non-exclusive, no-charge, royalty-free, irrevocable copyright license to reproduce, prepare, publicly display, publicly perform, sublicense, and distribute the Complementary Material, the Model, and Derivatives of the Model.
# 3. Grant of Patent License. Subject to the terms and conditions of this License and where and as applicable, each Contributor hereby grants to You a perpetual, worldwide, non-exclusive, no-charge, royalty-free, irrevocable (except as stated in this paragraph) patent license to make, have made, use, offer to sell, sell, import, and otherwise transfer the Model and the Complementary Material, where such license applies only to those patent claims licensable by such Contributor that are necessarily infringed by their Contribution(s) alone or by combination of their Contribution(s) with the Model to which such Contribution(s) was submitted. If You institute patent litigation against any entity (including a cross-claim or counterclaim in a lawsuit) alleging that the Model and/or Complementary Material or a Contribution incorporated within the Model and/or Complementary Material constitutes direct or contributory patent infringement, then any patent licenses granted to You under this License for the Model and/or Work shall terminate as of the date such litigation is asserted or filed.

# Section III: CONDITIONS OF USAGE, DISTRIBUTION AND REDISTRIBUTION

# 4. Distribution and Redistribution. You may host for Third Party remote access purposes (e.g. software-as-a-service), reproduce and distribute copies of the Model or Derivatives of the Model thereof in any medium, with or without modifications, provided that You meet the following conditions:
# Use-based restrictions as referenced in paragraph 5 MUST be included as an enforceable provision by You in any type of legal agreement (e.g. a license) governing the use and/or distribution of the Model or Derivatives of the Model, and You shall give notice to subsequent users You Distribute to, that the Model or Derivatives of the Model are subject to paragraph 5. This provision does not apply to the use of Complementary Material.
# You must give any Third Party recipients of the Model or Derivatives of the Model a copy of this License;
# You must cause any modified files to carry prominent notices stating that You changed the files;
# You must retain all copyright, patent, trademark, and attribution notices excluding those notices that do not pertain to any part of the Model, Derivatives of the Model.
# You may add Your own copyright statement to Your modifications and may provide additional or different license terms and conditions - respecting paragraph 4.a. - for use, reproduction, or Distribution of Your modifications, or for any such Derivatives of the Model as a whole, provided Your use, reproduction, and Distribution of the Model otherwise complies with the conditions stated in this License.
# 5. Use-based restrictions. The restrictions set forth in Attachment A are considered Use-based restrictions. Therefore You cannot use the Model and the Derivatives of the Model for the specified restricted uses. You may use the Model subject to this License, including only for lawful purposes and in accordance with the License. Use may include creating any content with, finetuning, updating, running, training, evaluating and/or reparametrizing the Model. You shall require all of Your users who use the Model or a Derivative of the Model to comply with the terms of this paragraph (paragraph 5).
# 6. The Output You Generate. Except as set forth herein, Licensor claims no rights in the Output You generate using the Model. You are accountable for the Output you generate and its subsequent uses. No use of the output can contravene any provision as stated in the License.

# Section IV: OTHER PROVISIONS

# 7. Updates and Runtime Restrictions. To the maximum extent permitted by law, Licensor reserves the right to restrict (remotely or otherwise) usage of the Model in violation of this License, update the Model through electronic means, or modify the Output of the Model based on updates. You shall undertake reasonable efforts to use the latest version of the Model.
# 8. Trademarks and related. Nothing in this License permits You to make use of Licensors’ trademarks, trade names, logos or to otherwise suggest endorsement or misrepresent the relationship between the parties; and any rights not expressly granted herein are reserved by the Licensors.
# 9. Disclaimer of Warranty. Unless required by applicable law or agreed to in writing, Licensor provides the Model and the Complementary Material (and each Contributor provides its Contributions) on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied, including, without limitation, any warranties or conditions of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A PARTICULAR PURPOSE. You are solely responsible for determining the appropriateness of using or redistributing the Model, Derivatives of the Model, and the Complementary Material and assume any risks associated with Your exercise of permissions under this License.
# 10. Limitation of Liability. In no event and under no legal theory, whether in tort (including negligence), contract, or otherwise, unless required by applicable law (such as deliberate and grossly negligent acts) or agreed to in writing, shall any Contributor be liable to You for damages, including any direct, indirect, special, incidental, or consequential damages of any character arising as a result of this License or out of the use or inability to use the Model and the Complementary Material (including but not limited to damages for loss of goodwill, work stoppage, computer failure or malfunction, or any and all other commercial damages or losses), even if such Contributor has been advised of the possibility of such damages.
# 11. Accepting Warranty or Additional Liability. While redistributing the Model, Derivatives of the Model and the Complementary Material thereof, You may choose to offer, and charge a fee for, acceptance of support, warranty, indemnity, or other liability obligations and/or rights consistent with this License. However, in accepting such obligations, You may act only on Your own behalf and on Your sole responsibility, not on behalf of any other Contributor, and only if You agree to indemnify, defend, and hold each Contributor harmless for any liability incurred by, or claims asserted against, such Contributor by reason of your accepting any such warranty or additional liability.
# 12. If any provision of this License is held to be invalid, illegal or unenforceable, the remaining provisions shall be unaffected thereby and remain valid as if such provision had not been set forth herein.

# END OF TERMS AND CONDITIONS


# Attachment A

# Use Restrictions

# You agree not to use the Model or Derivatives of the Model:
# - In any way that violates any applicable national, federal, state, local or international law or regulation;
# - For the purpose of exploiting, harming or attempting to exploit or harm minors in any way;
# - To generate or disseminate verifiably false information and/or content with the purpose of harming others;
# - To generate or disseminate personal identifiable information that can be used to harm an individual;
# - To defame, disparage or otherwise harass others;
# - For fully automated decision making that adversely impacts an individual’s legal rights or otherwise creates or modifies a binding, enforceable obligation;
# - For any use intended to or which has the effect of discriminating against or harming individuals or groups based on online or offline social behavior or known or predicted personal or personality characteristics;
# - To exploit any of the vulnerabilities of a specific group of persons based on their age, social, physical or mental characteristics, in order to materially distort the behavior of a person pertaining to that group in a manner that causes or is likely to cause that person or another person physical or psychological harm;
# - For any use intended to or which has the effect of discriminating against individuals or groups based on legally protected characteristics or categories;
# - To provide medical advice and medical results interpretation;
# - To generate or disseminate information for the purpose to be used for administration of justice, law enforcement, immigration or asylum processes, such as predicting an individual will commit fraud/crime commitment (e.g. by text profiling, drawing causal relationships between assertions made in documents, indiscriminate and arbitrarily-targeted use).

import argparse
import os
import sys
import datetime
import glob
import numpy as np
import time
import torch
import torchvision
import pytorch_lightning as pl
from pathlib import Path
import preprocessing

from packaging import version
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset
from functools import partial
from PIL import Image

from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import Callback, LearningRateMonitor
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.utilities import rank_zero_info
import wandb

from ldm.data.base import Txt2ImgIterableBaseDataset
from ldm.util import instantiate_from_config


def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="postfix for logdir",
    )
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="resume from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "-rc",
        "--resume-from-checkpoint-custom",
        type=str,
        const=True,
        default="../assets/pretrained_models/sd-v1-4.ckpt",
        nargs="?",
        help="resume from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "--delta_ckpt",
        type=str,
        const=True,
        default=None,
        nargs="?",
        help="resume from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
             "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list(),
    )
    parser.add_argument(
        "-t",
        "--train",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="train",
    )
    parser.add_argument(
        "--no-test",
        type=str2bool,
        const=True,
        default=True,
        nargs="?",
        help="disable test",
    )
    parser.add_argument(
        "-p",
        "--project",
        help="name of new or path to existing project"
    )
    parser.add_argument(
        "-d",
        "--debug",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="enable post-mortem debugging",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=23,
        help="seed for seed_everything",
    )
    parser.add_argument(
        "-f",
        "--postfix",
        type=str,
        default="",
        help="post-postfix for default name",
    )
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        default="logs",
        help="directory for logging dat shit",
    )
    parser.add_argument(
        "--scale_lr",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="scale base-lr by ngpu * batch_size * n_accumulate",
    )
    parser.add_argument(
        "--datapath",
        type=str,
        default="",
        help="path to target images",
    )
    parser.add_argument(
        "--reg_datapath",
        type=str,
        default=None,
        help="path to regularization images",
    )
    parser.add_argument(
        "--caption",
        type=str,
        default="",
        help="path to target images",
    )
    parser.add_argument(
        "--reg_caption",
        type=str,
        default="",
        help="path to target images",
    )
    parser.add_argument(
        "--datapath2",
        type=str,
        default="",
        help="path to target images",
    )
    parser.add_argument(
        "--reg_datapath2",
        type=str,
        default=None,
        help="path to regularization images",
    )
    parser.add_argument(
        "--caption2",
        type=str,
        default="",
        help="path to target images",
    )
    parser.add_argument(
        "--reg_caption2",
        type=str,
        default="",
        help="path to regularization images' caption",
    )
    parser.add_argument(
        "--modifier_token",
        type=str,
        default=None,
        help="token added before cateogry word for personalization use case",
    )
    parser.add_argument(
        "--freeze_model",
        type=str,
        default=None,
        help="crossattn to enable fine-tuning of all key, value, query matrices",
    )
    parser.add_argument(
        "--loss_type_reverse",
        type=str,
        default='model-based',
        help="loss type for reverse fine-tuning",
    )
    parser.add_argument(
        "--caption_target",
        type=str,
        required=True,
        help="target style to remove, used when kldiv loss",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=0,
        help="repeat the target dataset by how many times. Used when training without regularization",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="overwrite batch size",
    )
    parser.add_argument(
        "--base_lr",
        type=float,
        default=None,
        help="overwrite base learning rate"
    )
    parser.add_argument(
        "--save_freq",
        type=int,
        default=None,
        help='overwrite every_n_train_steps in model saving callback'
    )
    parser.add_argument(
        "--image_logging_freq",
        type=int,
        default=None,
        help='overwrite batch_frequency in image logging callback'
    )
    parser.add_argument(
        "--train_max_steps",
        type=int,
        default=None,
        help='overwrite max_steps in finetuning'
    )
    parser.add_argument(
        "--parameter_group",
        type=str,
        default=None,
        choices=['full-weight', 'cross-attn', 'embedding'],
        help='parameter groups to finetune. Default: full-weight for memorization and cross-attn for others'
    )
    parser.add_argument(
        "--concept_type",
        type=str,
        required=True,
        choices=['style', 'object', 'memorization'],
        help='the type of removed concepts'
    )
    parser.add_argument(
        "--prompts",
        type=str,
        default=None,
        help='the initial path to ablation prompts'
    )
    parser.add_argument(
        "--train_size",
        type=int,
        default=1000,
        help='the number of generated images'
    )
    parser.add_argument(
        "--root",
        type=str,
        default='data/',
        help='the root folder of generated training images'
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=10,
        help='number of batch size in image generation'
    )
    parser.add_argument(
        "--regularization",
        action='store_true',
        help='If True, add regularization loss'
    )
    parser.add_argument(
        "--mem_impath",
        type=str,
        default="",
        help='the path to saved memorized image'
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default="",
        help='the entity name for wandb logging'
    )
    return parser


def nondefault_trainer_args(opt):
    parser = argparse.ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args([])
    return sorted(k for k in vars(args) if getattr(opt, k) != getattr(args, k))


class WrappedDataset(Dataset):
    """Wraps an arbitrary object with __len__ and __getitem__ into a pytorch dataset"""

    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def worker_init_fn(_):
    worker_info = torch.utils.data.get_worker_info()

    dataset = worker_info.dataset
    worker_id = worker_info.id

    if isinstance(dataset, Txt2ImgIterableBaseDataset):
        split_size = dataset.num_records // worker_info.num_workers
        # reset num_records to the true number to retain reliable length information
        dataset.sample_ids = dataset.valid_ids[worker_id * split_size:(worker_id + 1) * split_size]
        current_id = np.random.choice(len(np.random.get_state()[1]), 1)
        return np.random.seed(np.random.get_state()[1][current_id] + worker_id)
    else:
        return np.random.seed(np.random.get_state()[1][0] + worker_id)


class ConcatDataset(Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, idx):
        return tuple(d[idx] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)


class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, batch_size, train=None, train2=None, validation=None, test=None, predict=None,
                 wrap=False, num_workers=None, shuffle_test_loader=False, use_worker_init_fn=False,
                 shuffle_val_dataloader=False):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_configs = dict()
        self.num_workers = num_workers if num_workers is not None else batch_size * 2
        self.use_worker_init_fn = use_worker_init_fn
        if train2 is not None and train2['params']['caption'] != '':
            self.dataset_configs["train2"] = train2
        if train is not None:
            self.dataset_configs["train"] = train
            self.train_dataloader = self._train_dataloader
        if validation is not None:
            self.dataset_configs["validation"] = validation
            self.val_dataloader = partial(self._val_dataloader, shuffle=shuffle_val_dataloader)
        if test is not None:
            self.dataset_configs["test"] = test
            self.test_dataloader = partial(self._test_dataloader, shuffle=shuffle_test_loader)
        if predict is not None:
            self.dataset_configs["predict"] = predict
            self.predict_dataloader = self._predict_dataloader
        self.wrap = wrap

    def prepare_data(self):
        for data_cfg in self.dataset_configs.values():
            instantiate_from_config(data_cfg)

    def setup(self, stage=None):
        self.datasets = dict(
            (k, instantiate_from_config(self.dataset_configs[k]))
            for k in self.dataset_configs)
        if self.wrap:
            for k in self.datasets:
                self.datasets[k] = WrappedDataset(self.datasets[k])

    def _train_dataloader(self):
        is_iterable_dataset = isinstance(self.datasets['train'], Txt2ImgIterableBaseDataset)
        if is_iterable_dataset or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        if "train2" in self.dataset_configs and self.dataset_configs["train2"]['params']["caption"] != '':
            train_set = self.datasets["train"]
            train2_set = self.datasets["train2"]
            concat_dataset = ConcatDataset(train_set, train2_set)
            return DataLoader(concat_dataset, batch_size=self.batch_size // 2,
                              num_workers=self.num_workers, shuffle=False if is_iterable_dataset else True,
                              worker_init_fn=init_fn)
        else:
            return DataLoader(self.datasets["train"], batch_size=self.batch_size,
                              num_workers=self.num_workers, shuffle=False if is_iterable_dataset else True,
                              worker_init_fn=init_fn)

    def _val_dataloader(self, shuffle=False):
        if isinstance(self.datasets['validation'], Txt2ImgIterableBaseDataset) or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(self.datasets["validation"],
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          worker_init_fn=init_fn,
                          shuffle=shuffle)

    def _test_dataloader(self, shuffle=False):
        is_iterable_dataset = isinstance(self.datasets['train'], Txt2ImgIterableBaseDataset)
        if is_iterable_dataset or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None

        # do not shuffle dataloader for iterable dataset
        shuffle = shuffle and (not is_iterable_dataset)

        return DataLoader(self.datasets["test"], batch_size=self.batch_size,
                          num_workers=self.num_workers, worker_init_fn=init_fn, shuffle=shuffle)

    def _predict_dataloader(self, shuffle=False):
        if isinstance(self.datasets['predict'], Txt2ImgIterableBaseDataset) or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(self.datasets["predict"], batch_size=self.batch_size,
                          num_workers=self.num_workers, worker_init_fn=init_fn)


class SetupCallback(Callback):
    def __init__(self, resume, now, logdir, ckptdir, cfgdir, config, lightning_config):
        super().__init__()
        self.resume = resume
        self.now = now
        self.logdir = logdir
        self.ckptdir = ckptdir
        self.cfgdir = cfgdir
        self.config = config
        self.lightning_config = lightning_config

    def on_keyboard_interrupt(self, trainer, pl_module):
        if trainer.global_rank == 0:
            print("Summoning checkpoint.")
            ckpt_path = os.path.join(self.ckptdir, "last.ckpt")
            trainer.save_checkpoint(ckpt_path)

    def on_fit_start(self, trainer, pl_module):
        if trainer.global_rank == 0:
            # Create logdirs and save configs
            os.makedirs(self.logdir, exist_ok=True)
            os.makedirs(self.ckptdir, exist_ok=True)
            os.makedirs(self.cfgdir, exist_ok=True)

            if "callbacks" in self.lightning_config:
                if 'metrics_over_trainsteps_checkpoint' in self.lightning_config['callbacks']:
                    os.makedirs(os.path.join(self.ckptdir, 'trainstep_checkpoints'), exist_ok=True)
            print("Project config")
            print(OmegaConf.to_yaml(self.config))
            OmegaConf.save(self.config,
                           os.path.join(self.cfgdir, "{}-project.yaml".format(self.now)))

            print("Lightning config")
            print(OmegaConf.to_yaml(self.lightning_config))
            OmegaConf.save(OmegaConf.create({"lightning": self.lightning_config}),
                           os.path.join(self.cfgdir, "{}-lightning.yaml".format(self.now)))

        else:
            # ModelCheckpoint callback created log directory --- remove it
            if not self.resume and os.path.exists(self.logdir):
                dst, name = os.path.split(self.logdir)
                dst = os.path.join(dst, "child_runs", name)
                os.makedirs(os.path.split(dst)[0], exist_ok=True)
                try:
                    os.rename(self.logdir, dst)
                except FileNotFoundError:
                    pass


class ImageLogger(Callback):
    def __init__(self, batch_frequency, max_images, save_freq=100, clamp=True, increase_log_steps=True,
                 rescale=True, disabled=False, log_on_batch_idx=False, log_first_step=False,
                 log_images_kwargs=None):
        super().__init__()
        self.rescale = rescale
        self.batch_freq = batch_frequency
        self.max_images = max_images
        self.save_freq = save_freq
        # self.logger_log_images = {
        #     pl.loggers.TensorBoardLogger: self._tb,
        #     pl.loggers.WandbLogger: self._wandb,
        # }

        self.log_steps = [2 ** n for n in range(int(np.log2(self.batch_freq)) + 1)]
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        self.log_first_step = log_first_step

    @rank_zero_only
    def log_local(self, logger, save_dir, split, images,
                  global_step, current_epoch, batch_idx):
        root = os.path.join(save_dir, "images", split)
        for k in images:
            grid = torchvision.utils.make_grid(images[k], nrow=4)
            if self.rescale:
                grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
            grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
            grid = grid.numpy()
            grid = (grid * 255).astype(np.uint8)
            filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(
                k,
                global_step,
                current_epoch,
                batch_idx)
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            im = Image.fromarray(grid)
            im.save(path)
            if isinstance(logger, pl.loggers.TensorBoardLogger):
                logger.experiment.add_image(
                    f'{split}_{k}', torchvision.utils.make_grid(images[k], nrow=4), global_step)
            else:
                logger.log_image(
                    key=f'{split}_{k}', images=[path], step=global_step)

    def log_img(self, pl_module, batch, batch_idx, split="train"):
        check_idx = batch_idx if self.log_on_batch_idx else pl_module.global_step
        if (self.check_frequency(check_idx) and  # batch_idx % self.batch_freq == 0
                hasattr(pl_module, "log_images") and
                callable(pl_module.log_images) and
                self.max_images > 0):

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                if isinstance(batch, list):
                    images = pl_module.log_images(batch[0], split=split, **self.log_images_kwargs)
                    images1 = pl_module.log_images(batch[1], split=split, **self.log_images_kwargs)
                else:
                    images = pl_module.log_images(batch, split=split, **self.log_images_kwargs)

            for k in images:
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                    if self.clamp:
                        images[k] = torch.clamp(images[k], -1., 1.)
            if isinstance(batch, list):
                for k in images1:
                    N = min(images1[k].shape[0], self.max_images)
                    images1[k] = images1[k][:N]
                    if isinstance(images1[k], torch.Tensor):
                        images1[k] = images1[k].detach().cpu()
                        if self.clamp:
                            images1[k] = torch.clamp(images1[k], -1., 1.)

            self.log_local(pl_module.logger, pl_module.logger.save_dir, split, images,
                           pl_module.global_step, pl_module.current_epoch, batch_idx)

            if isinstance(batch, list):
                self.log_local(pl_module.logger, pl_module.logger.save_dir, split + "_reg", images1,
                               pl_module.global_step, pl_module.current_epoch, batch_idx)

            if is_train:
                pl_module.train()

    def check_frequency(self, check_idx):
        if ((check_idx % self.batch_freq) == 0 or (check_idx in self.log_steps)) and (
                check_idx > 0 or self.log_first_step):
            try:
                self.log_steps.pop(0)
            except IndexError as e:
                print(e)
                pass
            return True
        return False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not self.disabled and (pl_module.global_step > 0 or self.log_first_step):
            self.log_img(pl_module, batch, batch_idx, split="train")
        if self.save_freq is not None:
            global_step = trainer.global_step
            if global_step % self.save_freq == 0:
                filename = f'step_{global_step}.ckpt'
                ckpt_path = os.path.join(trainer.checkpoint_callback.dirpath, filename)
                trainer.save_checkpoint(ckpt_path)


class CUDACallback(Callback):
    # see https://github.com/SeanNaren/minGPT/blob/master/mingpt/callback.py
    def on_train_epoch_start(self, trainer, pl_module):
        # Reset the memory use counter
        torch.cuda.reset_peak_memory_stats(trainer.strategy.root_device.index)
        torch.cuda.synchronize(trainer.strategy.root_device.index)
        self.start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        torch.cuda.synchronize(trainer.strategy.root_device.index)
        max_memory = torch.cuda.max_memory_allocated(trainer.strategy.root_device.index) / 2 ** 20
        epoch_time = time.time() - self.start_time

        try:
            max_memory = trainer.training_type_plugin.reduce(max_memory)
            epoch_time = trainer.training_type_plugin.reduce(epoch_time)

            rank_zero_info(f"Average Epoch time: {epoch_time:.2f} seconds")
            rank_zero_info(f"Average Peak memory {max_memory:.2f}MiB")
        except AttributeError:
            pass

    def on_before_optimizer_step(self, trainer, pl_module, optimizer, opt_idx):
        if pl_module.freeze_model == 'none':
            # Get the index for tokens that we want to zero the grads for
            grads_text_encoder = pl_module.cond_stage_model.transformer.get_input_embeddings().weight.grad
            index_grads_to_zero = torch.arange(len(pl_module.cond_stage_model.tokenizer)) != \
                pl_module.cond_stage_model.modifier_token_id[0]
            for i in range(len(pl_module.cond_stage_model.modifier_token_id[1:])):
                index_grads_to_zero = index_grads_to_zero & (torch.arange(len(pl_module.cond_stage_model.tokenizer)) !=
                                                             pl_module.cond_stage_model.modifier_token_id[i])
            grads_text_encoder.data[index_grads_to_zero, :] = grads_text_encoder.data[index_grads_to_zero, :].fill_(0)


if __name__ == "__main__":
    torch.cuda.empty_cache()
    # custom parser to specify config files, train, test and debug mode,
    # postfix, resume.
    # `--key value` arguments are interpreted as arguments to the trainer.
    # `nested.key=value` arguments are interpreted as config parameters.
    # configs are merged from left-to-right followed by command line parameters.

    # model:
    #   base_learning_rate: float
    #   target: path to lightning module
    #   params:
    #       key: value
    # data:
    #   target: main.DataModuleFromConfig
    #   params:
    #      batch_size: int
    #      wrap: bool
    #      train:
    #          target: path to train dataset
    #          params:
    #              key: value
    #      validation:
    #          target: path to validation dataset
    #          params:
    #              key: value
    #      test:
    #          target: path to test dataset
    #          params:
    #              key: value
    # lightning: (optional, has sane defaults and can be specified on cmdline)
    #   trainer:
    #       additional arguments to trainer
    #   logger:
    #       logger to instantiate
    #   modelcheckpoint:
    #       modelcheckpoint to instantiate
    #   callbacks:
    #       callback1:
    #           target: importpath
    #           params:
    #               key: value

    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    # add cwd for convenience and to make classes in this file available when
    # running as `python main.py`
    # (in particular `main.DataModuleFromConfig`)
    sys.path.append(os.getcwd())

    parser = get_parser()
    parser = Trainer.add_argparse_args(parser)
    # import torch.multiprocessing
    # torch.multiprocessing.set_sharing_strategy('file_system')

    opt, unknown = parser.parse_known_args()
    if opt.name and opt.resume:
        raise ValueError(
            "-n/--name and -r/--resume cannot be specified both."
            "If you want to resume training in a new log folder, "
            "use -n/--name in combination with --resume_from_checkpoint"
        )

    if opt.concept_type == 'style':
        opt.base = ['configs/finetune_style.yaml']
    elif opt.concept_type == 'object':
        opt.base = ['configs/finetune_object.yaml']
    else:
        opt.base = ['configs/finetune_mem.yaml']

    if opt.resume:
        if not os.path.exists(opt.resume):
            raise ValueError("Cannot find {}".format(opt.resume))
        if os.path.isfile(opt.resume):
            paths = opt.resume.split("/")
            # idx = len(paths)-paths[::-1].index("logs")+1
            # logdir = "/".join(paths[:idx])
            logdir = "/".join(paths[:-2])
            ckpt = opt.resume
        else:
            assert os.path.isdir(opt.resume), opt.resume
            logdir = opt.resume.rstrip("/")
            ckpt = os.path.join(logdir, "checkpoints", "last.ckpt")

        opt.resume_from_checkpoint = ckpt
        base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*.yaml")))

        opt.base = base_configs + opt.base
        print(opt.base)
        _tmp = logdir.split("/")
        nowname = _tmp[-1]
    else:
        if opt.name:
            name = "_" + opt.name
        elif opt.base:
            cfg_fname = os.path.split(opt.base[0])[-1]
            cfg_name = os.path.splitext(cfg_fname)[0]
            name = "_" + cfg_name
        else:
            name = ""
        nowname = now + name + opt.postfix
        logdir = os.path.join(opt.logdir, nowname)

    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")
    seed_everything(opt.seed)

    try:
        # init and save configs
        configs = [OmegaConf.load(cfg) for cfg in opt.base]
        cli = OmegaConf.from_dotlist(unknown)
        config = OmegaConf.merge(*configs, cli)
        lightning_config = config.pop("lightning", OmegaConf.create())
        # merge trainer cli with config
        trainer_config = lightning_config.get("trainer", OmegaConf.create())
        # default to ddp
        trainer_config["accelerator"] = "gpu"
        for k in nondefault_trainer_args(opt):
            trainer_config[k] = getattr(opt, k)
        if not ("gpus" in trainer_config):
            del trainer_config["accelerator"]
            cpu = True
        else:
            gpuinfo = trainer_config["gpus"]
            del trainer_config["gpus"]
            trainer_config["devices"] = gpuinfo
            trainer_config["strategy"] = "ddp"
            print(f"Running on GPUs {gpuinfo}")
            cpu = False
        if opt.train_max_steps is not None:
            trainer_config.max_steps = opt.train_max_steps
        trainer_opt = argparse.Namespace(**trainer_config)
        lightning_config.trainer = trainer_config

        # generating images if prompts are provided
        if opt.prompts is None:
            if opt.datapath == '' or opt.caption == '':
                print('either initial prompts or path to generated images folder should be provided')
                raise NotImplementedError
        else:
            name = Path(opt.prompts).stem
            gen_folder = Path(opt.root) / (name + '_gen')
            if not gen_folder.exists():
                gen_folder.mkdir()
            # create prompts
            ranks = [int(i) for i in trainer_config["devices"].split(',') if i != ""]
            preprocessing.preprocess(opt, opt.prompts, gen_folder, opt.concept_type, ranks)
            opt.datapath = str(gen_folder / 'images.txt')
            opt.caption = str(gen_folder / 'caption.txt')
            if opt.regularization:
                opt.datapath2 = str(gen_folder / 'images.txt')
                opt.caption2 = str(gen_folder / 'caption.txt')

        # data
        config.data.params.train.params.caption = opt.caption
        config.data.params.train.params.reg_caption = opt.reg_caption
        config.data.params.train.params.datapath = opt.datapath
        config.data.params.train.params.reg_datapath = opt.reg_datapath
        if opt.caption2 is not None:
            config.data.params.train2.params.caption = opt.caption2
            config.data.params.train2.params.reg_caption = opt.reg_caption2
            config.data.params.train2.params.datapath = opt.datapath2
            config.data.params.train2.params.reg_datapath = opt.reg_datapath2

        # concept type and parameter group
        if opt.parameter_group is None:
            if opt.concept_type == 'memorization':
                opt.parameter_group = 'full-weight'
            else:
                opt.parameter_group = 'cross-attn'

        if opt.parameter_group == 'full-weight':
            config.model.params.cond_stage_trainable = False
            config.model.params.freeze_model = "all"
        elif opt.parameter_group == 'cross-attn':
            config.model.params.cond_stage_trainable = False
            config.model.params.freeze_model = "crossattn-kv"
        else:
            lightning_config.modelcheckpoint.params.every_n_train_steps = lightning_config.callbacks.image_logger.params.save_freq
            lightning_config.callbacks.image_logger.params.save_freq = 10000
            if opt.concept_type == 'memorization':
                print("embedding finetuning is not supported for memorization")
                raise NotImplementedError
            config.model.params.cond_stage_trainable = True
            config.model.params.freeze_model = "none"
            config.model.params.add_token = False

        config.data.params.validation = config.data.params.train
        if opt.batch_size is not None:
            config.data.params.batch_size = opt.batch_size
        if opt.base_lr is not None:
            config.model.base_learning_rate = opt.base_lr
        if opt.save_freq is not None:
            if opt.parameter_group == 'embedding':
                lightning_config.modelcheckpoint.params.every_n_train_steps = opt.save_freq
            else:
                lightning_config.callbacks.image_logger.params.save_freq = opt.save_freq
        if opt.image_logging_freq is not None:
            lightning_config.callbacks.image_logger.params.batch_frequency = opt.image_logging_freq
        if opt.train_max_steps is not None:
            lightning_config.trainer.max_steps = opt.train_max_steps
        if opt.modifier_token is not None:
            config.model.params.cond_stage_config.params.modifier_token = opt.modifier_token
        if opt.repeat > 0:
            config.data.params.train.params.repeat = opt.repeat

        # note: will make it consistent later
        if opt.caption_target is not None:
            if opt.caption2 == "":
                config.data.params.train.params.caption_target = opt.caption_target
            else:
                config.data.params.train2.params.caption_target = opt.caption_target

        if opt.resume_from_checkpoint_custom:
            config.model.params.ckpt_path = None
        if opt.freeze_model is not None:
            config.model.params.freeze_model = opt.freeze_model
        config.model.params.loss_type_reverse = opt.loss_type_reverse

        model = instantiate_from_config(config.model)

        if opt.resume_from_checkpoint_custom:
            st = torch.load(opt.resume_from_checkpoint_custom, map_location='cpu')["state_dict"]
            token_weights = st["cond_stage_model.transformer.text_model.embeddings.token_embedding.weight"]
            del st["cond_stage_model.transformer.text_model.embeddings.token_embedding.weight"]
            model.load_state_dict(st, strict=False)
            model.cond_stage_model.transformer.text_model.embeddings.token_embedding.weight.data[
                :token_weights.shape[0]] = token_weights

        if opt.delta_ckpt is not None:
            st = torch.load(opt.delta_ckpt)
            embed = None
            if 'embed' in st['state_dict']:
                embed = st['state_dict']['embed'].reshape(-1, 768)
            print("restroting from delta model from previous version")
            # st1 = model.state_dict()
            # for each in st1.keys():
            #     if each in st['state_dict'].keys():
            #         print("found common", each)
            model.load_state_dict(st['state_dict'], strict=False)
            if embed is not None:
                print(f"restoring embedding. Embedding shape: {embed.shape[0]}")
                model.cond_stage_model.transformer.text_model.embeddings.token_embedding.weight.data[
                    -embed.shape[0]:] = embed

        # trainer and callbacks
        trainer_kwargs = dict()

        # default logger configs
        default_logger_cfgs = {
            "wandb": {
                "target": "pytorch_lightning.loggers.WandbLogger",
                "params": {
                    "project": "concept-ablation",
                    "name": nowname,
                    "save_dir": logdir,
                    "dir": logdir,
                    "id": nowname,
                    "resume": "allow",
                    "entity": opt.wandb_entity,
                }
            },
            "tensorboard": {
                "target": "pytorch_lightning.loggers.TensorBoardLogger",
                "params": {
                    "name": "tensorboard",
                    "save_dir": logdir,
                }
            },
        }
        os.makedirs(logdir, exist_ok=True)
        if opt.wandb_entity != "":
            default_logger_cfg = default_logger_cfgs["wandb"]
        else:
            default_logger_cfg = default_logger_cfgs["tensorboard"]
        if "logger" in lightning_config:
            logger_cfg = lightning_config.logger
        else:
            logger_cfg = OmegaConf.create()
        logger_cfg = OmegaConf.merge(default_logger_cfg, logger_cfg)
        trainer_kwargs["logger"] = instantiate_from_config(logger_cfg)

        # modelcheckpoint - use TrainResult/EvalResult(checkpoint_on=metric) to
        # specify which metric is used to determine best models
        default_modelckpt_cfg = {
            "target": "pytorch_lightning.callbacks.ModelCheckpoint",
            "params": {
                "dirpath": ckptdir,
                "filename": "{epoch:06}",
                "verbose": True,
                "save_last": True,
            }
        }
        if hasattr(model, "monitor"):
            print(f"Monitoring {model.monitor} as checkpoint metric.")
            default_modelckpt_cfg["params"]["monitor"] = model.monitor
            default_modelckpt_cfg["params"]["save_top_k"] = -1

        if "modelcheckpoint" in lightning_config:
            modelckpt_cfg = lightning_config.modelcheckpoint
        else:
            modelckpt_cfg = OmegaConf.create()
        modelckpt_cfg = OmegaConf.merge(default_modelckpt_cfg, modelckpt_cfg)
        print(f"Merged modelckpt-cfg: \n{modelckpt_cfg}")
        if version.parse(pl.__version__) < version.parse('1.4.0'):
            trainer_kwargs["checkpoint_callback"] = instantiate_from_config(modelckpt_cfg)

        # add callback which sets up log directory
        default_callbacks_cfg = {
            "setup_callback": {
                "target": "train.SetupCallback",
                "params": {
                    "resume": opt.resume,
                    "now": now,
                    "logdir": logdir,
                    "ckptdir": ckptdir,
                    "cfgdir": cfgdir,
                    "config": config,
                    "lightning_config": lightning_config,
                }
            },
            "image_logger": {
                "target": "train.ImageLogger",
                "params": {
                    "batch_frequency": 750,
                    "max_images": 4,
                    "clamp": True
                }
            },
            "learning_rate_logger": {
                "target": "train.LearningRateMonitor",
                "params": {
                    "logging_interval": "step",
                    # "log_momentum": True
                }
            },
            "cuda_callback": {
                "target": "train.CUDACallback"
            },
        }
        if version.parse(pl.__version__) >= version.parse('1.4.0'):
            default_callbacks_cfg.update({'checkpoint_callback': modelckpt_cfg})

        if "callbacks" in lightning_config:
            callbacks_cfg = lightning_config.callbacks
        else:
            callbacks_cfg = OmegaConf.create()

        if 'metrics_over_trainsteps_checkpoint' in callbacks_cfg:
            print(
                'Caution: Saving checkpoints every n train steps without deleting. This might require some free space.')
            default_metrics_over_trainsteps_ckpt_dict = {
                'metrics_over_trainsteps_checkpoint':
                    {"target": 'pytorch_lightning.callbacks.ModelCheckpoint',
                     'params': {
                         "dirpath": os.path.join(ckptdir, 'trainstep_checkpoints'),
                         "filename": "{epoch:06}-{step:09}",
                         "verbose": True,
                         'save_top_k': -1,
                         'every_n_train_steps': modelckpt_cfg.param.every_n_train_steps,
                         'save_weights_only': True
                     }
                     }
            }
            default_callbacks_cfg.update(default_metrics_over_trainsteps_ckpt_dict)

        callbacks_cfg = OmegaConf.merge(default_callbacks_cfg, callbacks_cfg)
        if 'ignore_keys_callback' in callbacks_cfg and hasattr(trainer_opt, 'resume_from_checkpoint'):
            callbacks_cfg.ignore_keys_callback.params['ckpt_path'] = trainer_opt.resume_from_checkpoint
        elif 'ignore_keys_callback' in callbacks_cfg:
            del callbacks_cfg['ignore_keys_callback']

        trainer_kwargs["callbacks"] = [instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg]

        trainer = Trainer.from_argparse_args(trainer_opt, **trainer_kwargs)
        trainer.logdir = logdir
        # data
        data = instantiate_from_config(config.data)
        # NOTE according to https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html
        # calling these ourselves should not be necessary but it is.
        # lightning still takes care of proper multiprocessing though
        data.prepare_data()
        data.setup()
        print("#### Data #####")
        for k in data.datasets:
            print(f"{k}, {data.datasets[k].__class__.__name__}, {len(data.datasets[k])}")

        # configure learning rate
        bs, base_lr = config.data.params.batch_size, config.model.base_learning_rate
        if not cpu:
            ngpu = len(lightning_config.trainer.devices.strip(",").split(','))
        else:
            ngpu = 1
        if 'accumulate_grad_batches' in lightning_config.trainer:
            accumulate_grad_batches = lightning_config.trainer.accumulate_grad_batches
        else:
            accumulate_grad_batches = 1
        print(f"accumulate_grad_batches = {accumulate_grad_batches}")
        lightning_config.trainer.accumulate_grad_batches = accumulate_grad_batches
        if opt.scale_lr:
            model.learning_rate = accumulate_grad_batches * ngpu * bs * base_lr
            print(
                "Setting learning rate to {:.2e} = {} (accumulate_grad_batches) * {} (num_gpus) * {} (batchsize) * {:.2e} (base_lr)".format(
                    model.learning_rate, accumulate_grad_batches, ngpu, bs, base_lr))
        else:
            model.learning_rate = base_lr
            print("++++ NOT USING LR SCALING ++++")
            print(f"Setting learning rate to {model.learning_rate:.2e}")

        # allow checkpointing via USR1

        def melk(*args, **kwargs):
            # run all checkpoint hooks
            if trainer.global_rank == 0:
                print("Summoning checkpoint.")
                ckpt_path = os.path.join(ckptdir, "last.ckpt")
                trainer.save_checkpoint(ckpt_path)

        def divein(*args, **kwargs):
            if trainer.global_rank == 0:
                import pudb
                pudb.set_trace()

        import signal

        signal.signal(signal.SIGUSR1, melk)
        signal.signal(signal.SIGUSR2, divein)

        # run
        if opt.train:
            try:
                # trainer.logger.watch(model.model.diffusion_model, log_freq=10)
                trainer.fit(model, data)
            except Exception:
                melk()
                raise
        if not opt.no_test and not trainer.interrupted:
            trainer.test(model, data)
        wandb.finish()
    except Exception:
        if opt.debug and trainer.global_rank == 0:
            try:
                import pudb as debugger
            except ImportError:
                import pdb as debugger
            debugger.post_mortem()
        raise
    finally:
        # move newly created debug project to debug_runs
        if opt.debug and not opt.resume and trainer.global_rank == 0:
            dst, name = os.path.split(logdir)
            dst = os.path.join(dst, "debug_runs", name)
            os.makedirs(os.path.split(dst)[0], exist_ok=True)
            os.rename(logdir, dst)
        if trainer.global_rank == 0:
            print(trainer.profiler.summary())
