# This code is built from the Stable Diffusion repository: https://github.com/CompVis/stable-diffusion.
# Copyright (c) 2022 Robin Rombach and Patrick Esser and contributors.
# CreativeML Open RAIL-M
#
# ==========================================================================================
#
# Adobe’s modifications are Copyright 2022 Adobe Research. All rights reserved.
# Adobe’s modifications are licensed under the Adobe Research License. To view a copy of the license, visit
# LICENSE.md.
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

import os, glob
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from einops import rearrange
from torchvision.utils import make_grid
from pytorch_lightning import seed_everything
from torch import autocast
from pathlib import Path
from io import BytesIO
import torch.multiprocessing as mp

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
import wandb


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
    model.cond_stage_model.transformer.text_model.embeddings.token_embedding.weight.data[:token_weights.shape[0]] = token_weights
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)
    model.eval()
    return model


def initialize(config, ckpt, delta_ckpt, seed=42):
    "initialize a model and sampler given checkpoing path"
    if delta_ckpt is not None:
        if len(glob.glob(os.path.join(delta_ckpt.split('checkpoints')[0], "configs/*.yaml"))) > 0:
            config = sorted(glob.glob(os.path.join(delta_ckpt.split('checkpoints')[0], "configs/*.yaml")))[-1]
    else:
        if len(glob.glob(
                os.path.join(ckpt.split('checkpoints')[0], "configs/*.yaml"))) > 0:
            config = sorted(
                glob.glob(os.path.join(ckpt.split('checkpoints')[0], "configs/*.yaml")))[
                -1]

    seed_everything(seed)
    config = OmegaConf.load(f"{config}")
    model = load_model_from_config(config, f"{ckpt}")

    device = torch.device('cuda')
    model = model.to(device)
    sampler = DDIMSampler(model)
    return model, sampler, device


def sample(data, model, sampler, outpath, ddim_steps=200, ddim_eta=1.0,
                  n_iter=1, scale=6, batch_size=10, shape=(4, 64, 64),
                  fixed_code=False, device=None, skip_save=False, skip_grid=True,
                  metadata=True, base_count=0, n_rows=5, wandb_log=False, ckptname='base', rank=None):
    """
        decoupled image sampling function, including saving, visualizing and wandb logging
    """
    sample_path = os.path.join(outpath, f"samples")
    if not Path(sample_path).exists():
        Path(sample_path).mkdir()

    if metadata:
        images_path = []
        captions = []
    start_code = None
    if fixed_code:
        start_code = torch.randn([batch_size, ] + list(shape), device=device)
    precision_scope = autocast
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                for prompts in tqdm(data, desc="data"):
                    all_samples = list()
                    for n in trange(n_iter, desc="Sampling"):
                        print(prompts[0])
                        uc = None
                        if scale != 1.0:
                            uc = model.get_learned_conditioning(batch_size * [""])
                        if isinstance(prompts, tuple):
                            prompts = list(prompts)
                        c = model.get_learned_conditioning(prompts)
                        # shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                        samples_ddim, _ = sampler.sample(S=ddim_steps,
                                                         conditioning=c,
                                                         batch_size=batch_size,
                                                         shape=list(shape),
                                                         verbose=False,
                                                         unconditional_guidance_scale=scale,
                                                         unconditional_conditioning=uc,
                                                         eta=ddim_eta,
                                                         x_T=start_code)
                        # print(samples_ddim.size())
                        x_samples_ddim = model.decode_first_stage(samples_ddim)
                        x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                        x_samples_ddim = x_samples_ddim.cpu()

                        if not skip_save:
                            for x_sample, caption in zip(x_samples_ddim, prompts):
                                x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                img = Image.fromarray(x_sample.astype(np.uint8))
                                img.save(os.path.join(sample_path, f"{base_count:05}.png"))
                                if metadata:
                                    images_path.append(os.path.join(sample_path, f"{base_count:05}.png"))
                                    captions.append(caption)
                                base_count += 1

                        if not skip_grid:
                            all_samples.append(x_samples_ddim)

                    if not skip_grid:
                        # additionally, save as grid
                        grid = torch.stack(all_samples, 0)
                        grid = rearrange(grid, 'n b c h w -> (n b) c h w')
                        grid = make_grid(grid, nrow=n_rows)

                        # to image
                        grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                        img = Image.fromarray(grid.astype(np.uint8))

                        img = img.convert('RGB')
                        prompt_name = "".join(ch for ch in prompts[0] if ch.isalpha() or ch.isspace())
                        prompt_name = prompt_name.replace(" ", "-")
                        file_prompt_name = prompt_name[:60]
                        img.save(os.path.join(outpath,
                                              f'{file_prompt_name}_{scale}_{ddim_steps}_{ddim_eta}.jpg'),
                                 quality=70)
                        if wandb_log:
                            out = BytesIO()
                            img.save(out, format='jpeg', quality=70)
                            img = Image.open(out)
                            wandb.log({f'{file_prompt_name}_{scale}_{ddim_steps}_{ddim_eta}.jpg': [
                                wandb.Image(img, caption=ckptname)]})

                            out.close()

    image_txt_path = ''
    caption_txt_path = ''
    if metadata:
        if rank is None:
            image_txt_path = os.path.join(outpath, 'images.txt')
            caption_txt_path = os.path.join(outpath, 'caption.txt')
        else:
            image_txt_path = os.path.join(outpath, f'images{rank}.txt')
            caption_txt_path = os.path.join(outpath, f'caption{rank}.txt')
        with open(image_txt_path, 'w') as f:
            for i in images_path:
                f.write(i + '\n')
        with open(caption_txt_path, 'w') as f:
            for i in captions:
                f.write(i + '\n')
    print('++++++++++++++++++++++++++++++++++++')
    print('+ Generation Finished ! ++++++++++++')
    print('++++++++++++++++++++++++++++++++++++')
    return image_txt_path, caption_txt_path


def sample_images(data, rank, config, ckpt, delta_ckpt, outpath, base_count, ddim_steps):
    torch.cuda.set_device(rank)
    model, sampler, device = initialize(config, ckpt, delta_ckpt)
    return sample(data, model, sampler, outpath, ddim_steps, base_count=base_count, rank=rank)


def distributed_sample_images(data, ranks, config, ckpt, delta_ckpt, outpath, ddim_steps=200):
    """
        data        : list of batch prompts (2-dim list)
        ranks       : list of available GPU-cards
        config      : the config file to load model
        ckpt        : the checkpoint path to model
        delta_ckpt  : the checkpoint path to delta model
        outpath     : the root folder to save images
        ddim_steps  : the ddim steps in generation
    """
    process_stack = []
    count = 0
    size = int(np.ceil(len(data) / len(ranks)))
    for i, local_rank in enumerate(ranks):
        cur_data = data[i*size:(i+1)*size]
        base_count = i*size * len(data[0])
        process = mp.Process(target=sample_images,
                             args=(cur_data, local_rank, config, ckpt, delta_ckpt, outpath, base_count, ddim_steps))
        process.start()
        process_stack.append(process)
        count += 1
        # wait for each process running

    for process in process_stack:
        process.join()
    # merge metadata
    images_path = []
    captions = []
    for local_rank in ranks:
        cur_image_txt_path = os.path.join(outpath, f'images{local_rank}.txt')
        cur_caption_txt_path = os.path.join(outpath, f'caption{local_rank}.txt')
        with open(cur_image_txt_path, 'r') as f:
            images_path += f.read().splitlines()
        with open(cur_caption_txt_path, 'r') as f:
            captions += f.read().splitlines()
        os.remove(cur_image_txt_path)
        os.remove(cur_caption_txt_path)

    image_txt_path = os.path.join(outpath, 'images.txt')
    caption_txt_path = os.path.join(outpath, 'caption.txt')
    with open(image_txt_path, 'w') as f:
        for i in images_path:
            f.write(i + '\n')
    with open(caption_txt_path, 'w') as f:
        for i in captions:
            f.write(i + '\n')


def safe_dir(dir):
    if not dir.exists():
        dir.mkdir()
    return dir
