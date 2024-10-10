# A Survey on conditional image synthesis with diffusion model

[![Awesome](media/badge.svg)](https://github.com/zju-pi/Awesome-Conditional-Diffusion-Models/tree/main) 
[![License: MIT](media/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![visitors](https://visitor-badge.laobi.icu/badge?page_id=zju-pi/Awesome-Conditional-Diffusion-Models/tree/main)](https://visitor-badge.laobi.icu/badge?page_id=zju-pi/Awesome-Conditional-Diffusion-Models/tree/main)

The repository is based on our survey [Conditional Image Synthesis with Diffusion Models: A Survey](https://arxiv.org/pdf/2409.19365)

Zheyuan Zhan, Defang Chen, Jian-Ping Mei, Zhenghe Zhao, Jiawei Chen, Chun Chen, Siwei Lyu, Fellow, IEEE and Can Wang

Zhejiang University,University at Buffalo, State University of New York,Zhejiang University of Technology

## Abstract

Conditional image synthesis based on user-specified requirements is a key component in creating complex visual content. In recent years, diffusion-based generative modeling has become a highly effective way for conditional image synthesis, leading to exponential growth in the literature. However, the complexity of diffusion-based modeling, the wide range of image synthesis tasks, and the diversity of conditioning mechanisms present significant challenges for researchers to keep up with rapid developments and understand the core concepts on this topic. In this survey, we categorize existing works based on how conditions are integrated into the two fundamental components of diffusion-based modeling, i.e., the denoising network and the sampling process. We specifically highlight the underlying principles, advantages, and potential challenges of various conditioning approaches in the training, re-purposing, and specialization stages to construct a desired denoising network. We also summarize six mainstream conditioning mechanisms in the essential sampling process. All discussions are centered around popular applications. Finally, we pinpoint some critical yet still open problems to be solved in the future and suggest some possible solutions.

## 🎉News!

📆2024-10-05: Our comprehensive survey paper, summarizing related methods published before October 1, 2024, is now available.

## 📄BibTeX

```
@article{zhan2024conditional,
  title={Conditional Image Synthesis with Diffusion Models: A Survey},
  author={Zhan, Zheyuan and Chen, Defang and Mei, Jian-Ping and Zhao, Zhenghe and Chen, Jiawei and Chen, Chun and Lyu, Siwei and Wang, Can},
  journal={arXiv preprint arXiv:2409.19365},
  year={2024}
}
```

## Contents
- [Overview](#Overview)
  - [Paper Structure](#Paper Structure)
  - [Conditional image synthesis tasks](#Conditional image synthesis tasks)
- [Papers](#Papers)
  - [Condition Integration in Denoising Networks](#condition-integration-in-denoising-networks)
    - [Condition Integration in the Training Stage](#Condition-Integration-in-the-Training-Stage)
      - [Conditional models for text-to-image (T2I)](#Conditional-models-for-text-to-image-(T2I))
      - [Conditional Models for Image Restoration](#Conditional-Models-for-Image-Restoration)
      - [Conditional Models for Other Synthesis Scenarios](#Conditional-Models-for-Other-Synthesis-Scenarios)
    - [Condition Integration in the Re-purposing Stage](#Condition-Integration-in-the-Re-purposing-Stage)
      - [Re-purposed Conditional Encoders](#Re-purposed-Conditional-Encoders)
      - [Condition Injection](#Condition-Injection)
      - [Backbone Fine-tuning](#Backbone-Fine-tuning)
    - [Condition Integration in the Specialization Stage](#Condition-Integration-in-the-Specialization-Stage)
      - [Condition Injection](#Condition-Injection)
      - [Testing-time Model Fine-Tuning](#Testing-time-Model-Fine-Tuning)
  - [Condition Integration in the Sampling Process](#Condition-Integration-in-the-Sampling-Process)
    - [Inversion](#Inversion)
    - [Attention Manipulation](#Attention-Manipulation)
    - [Noise Blending](#Noise-Blending)
    - [Revising Diffusion Process](#Revising-Diffusion-Process)   
    - [Guidance](#Guidance)
    - [Conditional Correction](#Conditional-Correction)

# Overview

In the two figures below, they respectively illustrate the DCIS taxonomy in this survey and the categorization of conditional image synthesis tasks.

## Paper Structure

![Conditional image synthesis with diffusion model](https://github.com/Szy12345-liv/A-Survey-on-conditional-image-synthesis-with-diffusion-model/blob/main/Images/Conditional%20image%20synthesis%20with%20diffusion%20model.png)

## Conditional image synthesis tasks

![tasks](https://github.com/zju-pi/Awesome-Conditional-Diffusion-Models/blob/main/Images/conditional%20image%20synthesis%20tasks.png)

# Papers

The date in the table represents the publication date of the first version of the paper on Arxiv.

## Condition Integration in Denoising Networks

This figure provides an examplar workflow to build desired denoising network for conditional synthesis tasks including text-to-image, visual signals to image and customization via these three condition integration stages.

![Workflow](https://github.com/zju-pi/Awesome-Conditional-Diffusion-Models/blob/main/Images/workflow.png)

### Condition Integration in the Training Stage

#### Conditional models for text-to-image (T2I)

| Title                                                        | Task          | Date    | Publication |
| ------------------------------------------------------------ | ------------- | ------- | ----------- |
| [**Vector quantized diffusion model for text-to-image synthesis**](https://arxiv.org/abs/2111.14822) | Text-to-image | 2021.11 | CVPR2022    |
| [**High-resolution image synthesis with latent diffusion models**](https://arxiv.org/abs/2112.10752) | Text-to-image | 2021.12 | CVPR2022    |
| [**GLIDE: towards photorealistic image generation and editing with text-guided diffusion models**](https://arxiv.org/abs/2112.10741) | Text-to-image | 2021.12 | ICML2022    |
| [**Hierarchical text-conditional image generation with CLIP latents**](https://arxiv.org/abs/2204.06125) | Text-to-image | 2022.4  | ARXIV2022   |
| [**Photorealistic text-to-image diffusion models with deep language understanding**](https://arxiv.org/abs/2205.11487) | Text-to-image | 2022.5  | NeurIPS2022 |
| [**ediffi: Text-to-image diffusion models with an ensemble of expert denoisers**](https://arxiv.org/abs/2211.01324) | Text-to-image | 2022.11 | ARXIV2022   |


#### Conditional Models for Image Restoration

| Title                                                        | Task                                                    | Date    | Publication        |
| ------------------------------------------------------------ | ------- | ------------------ | ------------------ |
| [**Srdiff: Single image super-resolution with diffusion probabilistic models**](https://arxiv.org/abs/2104.14951) | Image restoration | 2021.4  | Neurocomputing2022 |
| [**Image super-resolution via iterative refinement**](https://arxiv.org/abs/2104.07636) | Image restoration | 2021.4  | TPAMI2022          |
| [**Cascaded diffusion models for high fidelity image generation**](https://arxiv.org/abs/2106.15282) | Image restoration | 2021.5  | JMLR2022           |
| [**Palette: Image-to-image diffusion models**](https://arxiv.org/abs/2111.05826) | Image restoration | 2021.11 | SIGGRAPH2022       |
| [**Denoising diffusion probabilistic models for robust image super-resolution in the wild**](https://arxiv.org/abs/2302.07864) | Image restoration | 2023.2  | ARXIV2023          |
| [**Resdiff: Combining cnn and diffusion model for image super-resolution**](https://arxiv.org/abs/2303.08714) | Image restoration | 2023.3  | AAAI2024           |
| [**Low-light image enhancement with wavelet-based diffusion models**](https://arxiv.org/abs/2306.00306) | Image restoration | 2023.6  | TOG2023|
| [**Wavelet-based fourier information interaction with frequency diffusion adjustment for underwater image restoration**](https://arxiv.org/abs/2311.16845) | Image restoration | 2023.11 | CVPR2024         |
| [**Diffusion-based blind text image super-resolution**](https://arxiv.org/abs/2312.08886) | Image restoration | 2023.12 | CVPR2023           |
| [**Low-light image enhancement via clip-fourier guided wavelet diffusion**](https://arxiv.org/abs/2401.03788) | Image restoration | 2024.1  | ARXIV2024          |


#### Conditional Models for Other Synthesis Scenarios

| Title                                                        | Task                                                    | Date    | Publication              |
| ------------------------------------------------------------ | ------- | ------------------------ | ------------------------ |
| [**Diffusion autoencoders: Toward a meaningful and decodable representation**](https://arxiv.org/abs/2111.15640) | Novel conditional control | 2021.11 | CVPR2022                 |
| [**Semantic image synthesis via diffusion models**](https://arxiv.org/abs/2207.00050) | visual feature map | 2022.6  | ARXIV2022               |
| [**A novel unified conditional scorebased generative framework for multi-modal medical image completion**](https://arxiv.org/abs/2207.03430) | Medical image synthesis | 2022.7  | ARXIV2022                |
| [**A morphology focused diffusion probabilistic model for synthesis of histopathology images**](https://arxiv.org/abs/2209.13167) | Medical image synthesis | 2022.9  | WACV2023                 |
| [**Humandiffusion: a coarse-to-fine alignment diffusion framework for controllable text-driven person image generation**](https://arxiv.org/abs/2211.06235) | Visual signal to image | 2022.11 | ARXIV2022 |
| [**Diffusion-based scene graph to image generation with masked contrastive pre-training**](https://arxiv.org/abs/2211.11138) | Graph to image | 2022.11 | ARXIV2022                |
| [**Dolce: A model-based probabilistic diffusion framework for limited-angle ct reconstruction**](https://arxiv.org/abs/2211.12340) | Medical image synthesis | 2022.11 | ICCV2023                 |
| [**Zero-shot medical image translation via frequency-guided diffusion models**](https://arxiv.org/abs/2304.02742) | Image editing | 2023.4  | Trans. Med. Imaging 2023 |
| [**Learned representation-guided diffusion models for large-image generation**](https://arxiv.org/abs/2312.07330) | / | 2023.12 | ARXIV2023                |

### Condition Integration in the Re-purposing Stage

#### Re-purposed Conditional Encoders

| Title                                                        | Task                             | Date    | Publication  |
| ------------------------------------------------------------ | -------------------------------- | ------- | ------------ |
| [**Pretraining is all you need for image-to-image translation**](https://arxiv.org/abs/2205.12952) | Visual signal to image           | 2022.5  | ARXIV2022    |
| [**T2i-adapter: Learning adapters to dig out more controllable ability for text-to-image diffusion models**](https://arxiv.org/abs/2302.08453) | Visual signal to image           | 2023.2  | AAAI2024     |
| [**Adding conditional control to text-to-image diffusion models**](https://arxiv.org/abs/2302.05543) | Visual signal to image           | 2023.2  | ICCV2023     |
| [**Encoder-based domain tuning for fast personalization of text-to-image models**](https://arxiv.org/abs/2302.12228) | Customization                    | 2023.2  | TOG2023      |
| [**Pair-diffusion: Object-level image editing with structure-and-appearance paired diffusion models**](https://arxiv.org/abs/2303.17546v1) | Image editing, Image composition | 2023.3  | ARXIV2023    |
| [**Taming encoder for zero fine-tuning image customization with text-to-image diffusion models**](https://arxiv.org/abs/2304.02642) | Customization                    | 2023.4  | ARXIV2023    |
| [**Instantbooth: Personalized text-to-image generation without test-time finetuning**](https://arxiv.org/abs/2304.03411) | Customization                    | 2023.4  | CVPR2024     |
| [**Blip-diffusion: pre-trained subject representation for controllable text-to-image generation and editing**](https://arxiv.org/abs/2305.14720) | Customization                    | 2023.5  | NeurIPS2023  |
| [**Fastcomposer: Tuning-free multi-subject image generation with localized attention**](https://arxiv.org/abs/2305.10431) | Customization                    | 2023.5  | ARXIV2023    |
| [**Prompt-free diffusion: Taking” text” out of text-to-image diffusion models**](https://arxiv.org/abs/2305.16223) | Visual signal to image           | 2023.5  | CVPR2024     |
| [**Paste,inpaint and harmonize via denoising: Subject-driven image editing with pre-trained diffusion model**](https://arxiv.org/abs/2306.07596) | Image composition                | 2023.6  | ARXIV2023    |
| [**Subject-diffusion: Open domain personalized text-to-image generation without test-time fine-tuning**](https://arxiv.org/abs/2307.11410) | Customization,Layout control     | 2023.7  | SIGGRAPH2024 |
| [**Imagebrush: Learning visual in-context instructions for exemplar-based image manipulation**](https://arxiv.org/abs/2308.00906) | Image editing                    | 2023.8  | NeurIPS2024  |
| [**Guiding instruction-based image editing via multimodal large language models**](https://arxiv.org/abs/2309.17102) | Image editing                    | 2023.9  | ARXIV2023    |
| [**Ranni: Taming text-to-image diffusion for accurate instruction following**](https://arxiv.org/abs/2311.17002) | Image editing                    | 2023.11 | ARXIV2023    |
| [**Smartedit: Exploring complex instruction-based image editing with multimodal large language models**](https://arxiv.org/abs/2312.06739) | Image editing                    | 2023.12 | ARXIV2023    |
| [**Instructany2pix: Flexible visual editing via multimodal instruction following**](https://arxiv.org/abs/2312.06738) | Image editing                    | 2023.12 | ARXIV2023    |
| [**Warpdiffusion: Efficient diffusion model for high-fidelity virtual try-on**](https://arxiv.org/abs/2312.03667) | Image composition                | 2023.12 | ARXIV2023    |
| [**Coarse-to-fine latent diffusion for pose-guided person image synthesis**](https://arxiv.org/abs/2402.18078) | Customization                    | 2024.2  | CVPR2024     |
| [**Lightit: Illumination modeling and control for diffusion models**](https://arxiv.org/abs/2403.10615) | Visual signal to image           | 2024.3  | CVPR2024     |
| [**Face2diffusion for fast and editable face personalization**](https://arxiv.org/abs/2403.05094) | Customization                    | 2024.3  | CVPR2024     |

#### Condition Injection

| Title                                                        | Task                                 | Date    | Publication |
| ------------------------------------------------------------ | ------------------------------------ | ------- | ----------- |
| [**GLIGEN: open-set grounded text-to-image generation**](https://arxiv.org/abs/2301.07093) | Layout control                       | 2023.1  | CVPR2023    |
| [**Elite: Encoding visual concepts into textual embeddings for customized text-to-image generation**](https://arxiv.org/abs/2302.13848) | Customization                        | 2023.2  | CVPR2023    |
| [**Mix-of-show: Decentralized low-rank adaptation for multi-concept customization of diffusion models**](https://arxiv.org/abs/2305.18292) | Customization                        | 2023.5  | NeurIPS2024 |
| [**Dragondiffusion: Enabling drag-style manipulation on diffusion models**](https://arxiv.org/abs/2307.02421) | Image editing                        | 2023.7  | ICLR2024    |
| [**Ip-adapter: Text compatible image prompt adapter for text-to-image diffusion models**](https://arxiv.org/abs/2308.06721) | Visual signal to image,Image editing | 2023.8  | ARXIV2023   |
| [**Interactdiffusion: Interaction control in text-to-image diffusion models**](https://arxiv.org/abs/2312.05849) | Layout control                       | 2023.12 | ARXIV2023   |
| [**Instancediffusion: Instance-level control for image generation**](https://arxiv.org/abs/2402.03290) | Layout control                       | 2024.2  | CVPR2024    |
| [**Deadiff: An efficient stylization diffusion model with disentangled representations**](https://arxiv.org/abs/2403.06951) | Image editing                        | 2024.3  | CVPR2024    |

#### Backbone Fine-tuning

| Title                                                        |                   | Date    | Publication |
| ------------------------------------------------------------ | ----------------- | ------- | ----------- |
| [**Instructpix2pix: Learning to follow image editing instructions**](https://arxiv.org/abs/2211.09800) | Image editing     | 2022.11 | CVPR2023    |
| [**Paint by example: Exemplar-based image editing with diffusion models**](https://arxiv.org/abs/2211.13227) | Image composition | 2022.11 | CVPR2023    |
| [**Objectstitch: Object compositing with diffusion model**](https://arxiv.org/abs/2212.00932v1) | Image composition | 2022.12 | CVPR2023    |
| [**Smartbrush: Text and shape guided object inpainting with diffusion model**](https://arxiv.org/abs/2212.05034) | Image restoration | 2022.12 | CVPR2023    |
| [**Imagen editor and editbench: Advancing and evaluating text-guided image inpainting**](https://arxiv.org/abs/2212.06909) | Image restoration | 2022.12 | CVPR2023    |
| [**Reference-based image composition with sketch via structure-aware diffusion model**](https://arxiv.org/abs/2304.09748) | Image composition | 2023.3  | ARXIV2023   |
| [**Dialogpaint: A dialogbased image editing model**](https://arxiv.org/abs/2303.10073) | Image editing     | 2023.3  | ARXIV2023   |
| [**Hive: Harnessing human feedback for instructional visual editing**](https://arxiv.org/abs/2303.09618) | Image editing     | 2023.3  | CVPR2024    |
| [**Inst-inpaint: Instructing to remove objects with diffusion models**](https://arxiv.org/abs/2304.03246) | Image editing     | 2023.4  | ARXIV2023   |
| [**Text-to-image editing by image information removal**](https://arxiv.org/abs/2305.17489) | Image editing     | 2023.5  | WACV2024    |
| [**Magicbrush: A manually annotated dataset for instruction-guided image editing**](https://arxiv.org/abs/2306.10012) | Image editing     | 2023.6  | NeurIPS2024 |
| [**Anydoor: Zero-shot object-level image customization**](https://arxiv.org/abs/2307.09481) | Image composition | 2023.7  | CVPR2024    |
| [**Instructdiffusion: A generalist modeling interface for vision tasks**](https://arxiv.org/abs/2309.03895) | Image editing     | 2023.9  | ARXIV2023   |
| [**Emu edit: Precise image editing via recognition and generation tasks**](https://arxiv.org/abs/2311.10089) | Image editing     | 2023.11 | CVPR2024    |
| [**Dreaminpainter: Text-guided subject-driven image inpainting with diffusion models**](https://arxiv.org/abs/2312.03771) | Image composition | 2023.12 | ARXIV2023   |

### Condition Integration in the Specialization Stage

#### Conditional Projection

| Title                                                        | Task          | Date    | Publication |
| ------------------------------------------------------------ | ------------- | ------- | ----------- |
| [**An image is worth one word: Personalizing text-to-image generation using textual inversion**](https://arxiv.org/abs/2208.01618) | Customization | 2022.8  | ICLR2023    |
| [**Imagic: Text-based real image editing with diffusion models**](https://arxiv.org/abs/2210.09276) | Image editing | 2022.10 | CVPR2023    |
| [**Uncovering the disentanglement capability in text-to-image diffusion models**](https://arxiv.org/abs/2212.08698) | Image editing | 2022.12 | CVPR2023    |
| [**Preditor: Text guided image editing with diffusion prior**](https://arxiv.org/abs/2302.07979) | Image editing | 2023.2  | ARXIV2023   |
| [**iedit: Localised text-guided image editing with weak supervision**](https://arxiv.org/abs/2305.05947) | Image editing | 2023.5  | CVPR2024    |
| [**Forgedit: Text guided image editing via learning and forgetting**](https://arxiv.org/abs/2309.10556) | Image editing | 2023.9  | ARXIV2023   |
| [**Prompting hard or hardly prompting: Prompt inversion for text-to-image diffusion models**](https://arxiv.org/abs/2312.12416) | Image editing | 2023.12 | CVPR2024    |

#### Testing-time Model Fine-Tuning

| Title                                                        | task          | Date    | Publication       |
| ------------------------------------------------------------ | ------------- | ------- | ----------------- |
| [**Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation**](https://arxiv.org/abs/2208.12242) | Customization | 2022.8  | CVPR2023          |
| [**Imagic: Text-based real image editing with diffusion models**](https://arxiv.org/abs/2210.09276) | Image editing | 2022.10 | CVPR2023          |
| [**Unitune: Text-driven image editing by fine tuning a diffusion model on a single image**](https://arxiv.org/abs/2210.09477) | Image editing | 2022.10 | TOG2023           |
| [**Multi-concept customization of text-to-image diffusion**](https://arxiv.org/abs/2212.04488) | Customization | 2022.12 | CVPR2023          |
| [**Sine: Single image editing with text-to-image diffusion models**](https://arxiv.org/abs/2212.04489) | Image editing | 2022.12 | CVPR2023          |
| [**Encoder-based domain tuning for fast personalization of text-to-image models**](https://arxiv.org/abs/2302.12228) | Customization | 2023.2  | TOG2023           |
| [**Svdiff: Compact parameter space for diffusion fine-tuning**](https://arxiv.org/abs/2303.11305) | Customization | 2023.3  | ICCV2023          |
| [**Cones: concept neurons in diffusion models for customized generation**](https://arxiv.org/abs/2303.05125) | Customization | 2023.3  | ICML2023          |
| [**Custom-edit: Text-guided image editing with customized diffusion models**](https://arxiv.org/abs/2305.15779) | Customization | 2023.5  | ARXIV2023         |
| [**Mix-of-show: Decentralized low-rank adaptation for multi-concept customization of diffusion models**](https://arxiv.org/abs/2305.18292) | Customization | 2023.5  | NeurIPS2024       |
| [**Layerdiffusion: Layered controlled image editing with diffusion models**](https://arxiv.org/abs/2305.18676) | Image editing | 2023.5  | SIGGRAPH Asia2023 |
| [**Cones 2: Customizable image synthesis with multiple subjects**](https://arxiv.org/abs/2305.19327) | Customization | 2023.5  | NeurIPS2023       |

## Condition Integration in the Sampling Process

We illustrate six conditioning mechanisms with an exemplary image editing process in next figure.

![Sampling](https://github.com/zju-pi/Awesome-Conditional-Diffusion-Models/blob/main/Images/Sampling.png)

### Inversion

| Title                                                        | Task                                  | Date    | Publication |
| ------------------------------------------------------------ | ------------------------------------- | ------- | ----------- |
| [**Sdedit: Guided image synthesis and editing with stochastic differential equations**](https://arxiv.org/abs/2108.01073) | Image editing, Visual signal to image | 2021.8  | ICLR2022    |
| [**Dual diffusion implicit bridges for image-to-image translation**](https://arxiv.org/abs/2203.08382) | Image editing, Visual signal to image | 2022.3  | ICLR2023    |
| [**Null-text inversion for editing real images using guided diffusion models**](https://arxiv.org/abs/2211.09794) | Image editing                         | 2022.11 | CVPR2023    |
| [**Edict: Exact diffusion inversion via coupled transformations**](https://arxiv.org/abs/2211.12446) | Image editing                         | 2022.11 | CVPR2023    |
| [**A latent space of stochastic diffusion models for zero-shot image editing and guidance**](https://openaccess.thecvf.com/content/ICCV2023/papers/Wu_A_Latent_Space_of_Stochastic_Diffusion_Models_for_Zero-Shot_Image_ICCV_2023_paper.pdf) | Image editing                         | 2022.11 | ICCV2023    |
| [**Inversion-based style transfer with diffusion models**](https://arxiv.org/abs/2211.13203) | Image editing                         | 2022.11 | CVPR2023    |
| [**An edit friendly ddpm noise space: Inversion and manipulations**](https://arxiv.org/abs/2304.06140) | Image editing                         | 2023.4  | ARXIV2023   |
| [**Prompt tuning inversion for text-driven image editing using diffusion models**](https://arxiv.org/abs/2305.04441) | Image editing                         | 2023.5  | ICCV2023    |
| [**Negative-prompt inversion: Fast image inversion for editing with textguided diffusion models**](https://arxiv.org/abs/2305.16807) | Image editing                         | 2023.5  | ARXIV2023   |
| [**Dragdiffusion: Harnessing diffusion models for interactive point-based image editing**](https://arxiv.org/abs/2306.14435) | Image editing                         | 2023.6  | CVPR2024    |
| [**Tf-icon: Diffusion-based training-free cross-domain image composition**](https://arxiv.org/abs/2307.12493) | Image editing                         | 2023.7  | ICCV2023    |
| [**Stylediffusion: Controllable disentangled style transfer via diffusion models**](https://arxiv.org/abs/2308.07863) | Image editing                         | 2023.8  | ICCV2023    |
| [**Kv inversion: Kv embeddings learning for text-conditioned real image action editing**](https://arxiv.org/abs/2309.16608) | Image editing                         | 2023.9  | PRCV2023    |
| [**Effective real image editing with accelerated iterative diffusion inversion**](https://arxiv.org/abs/2309.04907) | Image editing                         | 2023.9  | ICCV2023    |
| [**Direct inversion: Boosting diffusion-based editing with 3 lines of code**](https://arxiv.org/abs/2310.01506) | Image editing                         | 2023.10 | ARXIV2023   |
| [**Ledits++: Limitless image editing using text-to-image models**](https://arxiv.org/abs/2311.16711) | Image editing                         | 2023.11 | CVPR2024    |
| [**The blessing of randomness: Sde beats ode in general diffusionbased image editing**](https://arxiv.org/abs/2311.01410) | Image editing                         | 2023.11 | ICLR2023    |
| [**Style injection in diffusion: A training-free approach for adapting large-scale diffusion models for style transfer**](https://arxiv.org/abs/2312.09008) | Image editing                         | 2023.12 | CVPR2024    |
| [**Fixed-point inversion for text-to-image diffusion models**](https://arxiv.org/abs/2312.12540v1) | Image editing                         | 2023.12 | ARXIV2023   |

### Attention Manipulation

| Title                                                        | Task           | Date    | Publication |
| ------------------------------------------------------------ | -------------- | ------- | ----------- |
| [**Prompt-to-prompt image editing with cross attention control**](https://arxiv.org/abs/2208.01626) | Image editing  | 2022.8  | ICLR2023    |
| [**Plug-and-play diffusion features for text-driven image-to-image translation**](https://arxiv.org/abs/2211.12572) | Image editing  | 2022.11 | CVPR2023    |
| [**ediffi: Text-toimage diffusion models with an ensemble of expert denoisers**](https://arxiv.org/abs/2211.01324) | Layout control | 2022.11 | ARXIV2022   |
| [**Masactrl: Tuning-free mutual self-attention control for consistent image synthesis and editing**](https://arxiv.org/abs/2304.08465) | Image editing  | 2023.4  | ICCV2023    |
| [**Custom-edit: Text-guided image editing with customized diffusion models**](https://arxiv.org/abs/2305.15779) | Customization  | 2023.5  | ARXIV2023   |
| [**Cones 2: Customizable image synthesis with multiple subjects**](https://arxiv.org/abs/2305.19327) | Customization  | 2023.5  | NeurIPS2023 |
| [**Dragdiffusion: Harnessing diffusion models for interactive point-based image editing**](https://arxiv.org/abs/2306.14435) | Image editing  | 2023.6  | CVPR2024    |
| [**Tf-icon: Diffusion-based training-free cross-domain image composition**](https://arxiv.org/abs/2307.12493) | Image editing  | 2023.7  | ICCV2023    |
| [**Dragondiffusion: Enabling drag-style manipulation on diffusion models**](https://arxiv.org/abs/2307.02421) | Image editing  | 2023.7  | ICLR2024    |
| [**Stylediffusion: Controllable disentangled style transfer via diffusion models**](https://arxiv.org/abs/2308.07863) | Image editing  | 2023.8  | ICCV2023    |
| [**Face aging via diffusion-based editing**](https://arxiv.org/abs/2309.11321) | Image editing  | 2023.9  | BMVC2023    |
| [**Dynamic prompt learning: Addressing cross-attention leakage for text-based image editing**](https://arxiv.org/abs/2309.15664) | Image editing  | 2023.9  | NeurIPS2024 |
| [**Style injection in diffusion: A training-free approach for adapting large-scale diffusion models for style transfer**](https://arxiv.org/abs/2312.09008) | Image editing  | 2023.12 | CVPR2024    |
| [**Focus on your instruction: Fine-grained and multi-instruction image editing by attention modulation**](https://arxiv.org/abs/2312.10113) | Image editing  | 2023.12 | ARXIV2023   |
| [**Towards understanding cross and self-attention in stable diffusion for text-guided image editing**](https://arxiv.org/abs/2403.03431) | Image editing  | 2024.3  | CVPR2024    |

### Noise Blending

| Title                                                        | Task                             | Date    | Publication |
| ------------------------------------------------------------ | -------------------------------- | ------- | ----------- |
| [**Compositional visual generation with composable diffusion models**](https://arxiv.org/abs/2206.01714) | General approach                 | 2022.6  | ECCV2022    |
| [**Classifier-free diffusion guidance**](https://arxiv.org/abs/2207.12598) | /                                | 2022.7  | ARXIV2022   |
| [**Sine: Single image editing with text-to-image diffusion models**](https://arxiv.org/abs/2212.04489) | Image editing                    | 2022.12 | CVPR2023    |
| [**Multidiffusion: Fusing diffusion paths for controlled image generation**](https://arxiv.org/abs/2302.08113) | Multiple control                 | 2023.2  | ICML2023    |
| [**Pair-diffusion: Object-level image editing with structure-and-appearance paired diffusion models**](https://arxiv.org/abs/2303.17546) | Image editing, Image composition | 2023.3  | ARXIV2023   |
| [**Magicfusion: Boosting text-to-image generation performance by fusing diffusion models**](https://arxiv.org/abs/2303.13126) | Image composition                | 2023.3  | ICCV2023    |
| [**Effective real image editing with accelerated iterative diffusion inversion**](https://arxiv.org/abs/2309.04907) | image editing                    | 2023.9  | ICCV2023    |
| [**Ledits++: Limitless image editing using text-to-image models**](https://arxiv.org/abs/2311.16711) | Image editing                    | 2023.11 | CVPR2024    |
| [**Noisecollage: A layout-aware text-to-image diffusion model based on noise cropping and merging**](https://arxiv.org/abs/2403.03485) | Image composition                | 2024.3  | CVPR2024    |

### Revising Diffusion Process

| Title                                                        | Task              | Date    | Publication |
| ------------------------------------------------------------ | ----------------- | ------- | ----------- |
| [**Snips: Solving noisy inverse problems stochastically**](https://arxiv.org/abs/2105.14951) | Image restoration | 2021.5  | NeurIPS2021 |
| [**Denoising diffusion restoration models**](https://arxiv.org/abs/2201.11793) | Image restoration | 2022.1  | NeurIPS2022 |
| [**Driftrec: Adapting diffusion models to blind jpeg restoration**](https://arxiv.org/abs/2211.06757) | Image restoration | 2022.11 | TIP2024     |
| [**Zero-shot image restoration using denoising diffusion null-space model**](https://arxiv.org/abs/2212.00490) | Image restoration | 2022.12 | ICLR2024    |
| [**Image restoration with mean-reverting stochastic differential equations**](https://arxiv.org/abs/2301.11699) | Image restoration | 2023.1  | ICML2023    |
| [**Inversion by direct iteration: An alternative to denoising diffusion for image restoration**](https://arxiv.org/abs/2303.11435) | Image restoration | 2023.3  | TMLR2023    |
| [**Resshift: Efficient diffusion model for image super-resolution by residual shifting**](https://arxiv.org/abs/2307.12348) | Image restoration | 2023.7  | NeurIPS2024 |
| [**Sinsr: diffusion-based image super-resolution in a single step**](https://arxiv.org/abs/2311.14760) | Image restoration | 2023.11 | CVPR2024    |

### Guidance

| Title                                                        | Task                                                    | Date    | Publication  |
| ------------------------------------------------------------ | ------- | ------------ | ------------ |
| [**Diffusion models beat gans on image synthesis**](https://arxiv.org/abs/2105.05233) | Text-to-image | 2021.5  | NeurIPS2021  |
| [**Blended diffusion for text-driven editing of natural images**](https://arxiv.org/abs/2111.14818) | Image restoration | 2021.11 | CVPR2022     |
| [**More control for free! image synthesis with semantic diffusion guidance**](https://arxiv.org/abs/2112.05744) | Text/Image-to-image | 2021.12 | WACV2023     |
| [**Improving diffusion models for inverse problems using manifold constraints**](https://arxiv.org/abs/2206.00941) | Image restoration | 2022.6  | NeurIPS2022  |
| [**Diffusion posterior sampling for general noisy inverse problems**](https://arxiv.org/abs/2209.14687) | Image restoration | 2022.9  | ICLR2023     |
| [**Diffusion-based image translation using disentangled style and content representation**](https://arxiv.org/abs/2209.15264) | Image editing | 2022.9  | ICLR2023     |
| [**Sketch-guided text-to-image diffusion models**](https://arxiv.org/abs/2211.13752) | Visual signal to image | 2022.11 | SIGGRAPH2023 |
| [**High-fidelity guided image synthesis with latent diffusion models**](https://arxiv.org/abs/2211.17084) | Visual signal to image | 2022.11 | CVPR2023     |
| [**Parallel diffusion models of operator and image for blind inverse problems**](https://arxiv.org/abs/2211.10656) | Image restoration | 2022.11 | CVPR2023     |
| [**Zero-shot image-to-image translation**](https://arxiv.org/abs/2302.03027) | Image editing | 2023.2  | SIGGRAPH2023 |
| [**Universal guidance for diffusion models**](https://arxiv.org/abs/2302.07121) | General guidance framework | 2023.2  | CVPR2023     |
| [**Pseudoinverse-guided diffusion models for inverse problems **](https://openreview.net/pdf?id=9_gsMA8MRKQ) | Image restoration | 2023.2  | ICLR2023     |
| [**Freedom: Training-free energy-guided conditional diffusion model**](https://arxiv.org/abs/2303.09833) | General guidance framework | 2023.3  | ICCV2023     |
| [**Training-free layout control with cross-attention guidance**](https://arxiv.org/abs/2304.03373) | Layout control | 2023.4  | WACV2024     |
| [**Generative diffusion prior for unified image restoration and enhancement**](https://arxiv.org/abs/2304.01247) | Image restoration | 2023.4  | CVPR2023  |
| [**Regeneration learning of diffusion models with rich prompts for zero-shot image translation**](https://arxiv.org/abs/2305.04651) | Image editing | 2023.5  | ARXIV2023    |
| [**Diffusion self-guidance for controllable image generation**](https://arxiv.org/abs/2306.00986) | Image editing | 2023.6  | NeurIPS2024  |
| [**Energy-based cross attention for bayesian context update in text-to-image diffusion models**](https://arxiv.org/abs/2306.09869) | Image editing | 2023.6  | NeurIPS2024  |
| [**Solving linear inverse problems provably via posterior sampling with latent diffusion models**](https://arxiv.org/abs/2307.00619) | Image restoration | 2023.7  | NeurIPS2024  |
| [**Dragondiffusion: Enabling drag-style manipulation on diffusion models**](https://arxiv.org/abs/2307.02421) | Image editing | 2023.7  | ICLR2024     |
| [**Readout guidance: Learning control from diffusion features**](https://arxiv.org/abs/2312.02150) | Visual signal to image | 2023.12 | CVPR2024     |
| [**Freecontrol: Training-free spatial control of any text-to-image diffusion model with any condition**](https://arxiv.org/abs/2312.07536) | Visual signal to image | 2023.12 | CVPR2024     |
| [**Diffeditor: Boosting accuracy and flexibility on diffusion-based image editing**](https://arxiv.org/abs/2402.02583) | Image editing | 2024.2  | CVPR2024     |



### Conditional Correction

| Title                                                        | Task              | Date    | Publication |
| ------------------------------------------------------------ | ----------------- | ------- | ----------- |
| [**Score-based generative modeling through stochastic differential equations**](https://arxiv.org/abs/2011.13456) | Image restoration | 2020.11 | ICLR2021    |
| [**ILVR: conditioning method for denoising diffusion probabilistic models**](https://arxiv.org/abs/2108.02938) | Image restoration | 2021.8  | ICCV2021    |
| [**Come-closer-diffuse-faster: Accelerating conditional diffusion models for inverse problems through stochastic contraction**](https://arxiv.org/abs/2112.05146) | Image restoration | 2021.12 | CVPR2022    |
| [**Repaint: Inpainting using denoising diffusion probabilistic models**](https://arxiv.org/abs/2201.09865) | Image restoration | 2022.1  | CVPR2022    |
| [**Improving diffusion models for inverse problems using manifold constraints**](https://arxiv.org/abs/2206.00941) | Image restoration | 2022.6  | NeurIPS2022 |
| [**Diffedit: Diffusion-based semantic image editing with mask guidance**](https://arxiv.org/abs/2210.11427) | Image editing     | 2022.10 | ICLR2023    |
| [**Region-aware diffusion for zero-shot text-driven image editing**](https://arxiv.org/abs/2302.11797) | Image editing     | 2023.2  | ARXIV2023   |
| [**Localizing object-level shape variations with text-to-image diffusion models**](https://arxiv.org/abs/2303.11306) | Image editing     | 2023.3  | ICCV2023    |
| [**Instructedit: Improving automatic masks for diffusion-based image editing with user instructions**](https://arxiv.org/abs/2305.18047) | Image editing     | 2023.5  | ARXIV2023   |
| [**Text-driven image editing via learnable regions**](https://arxiv.org/abs/2311.16432) | Image editing     | 2023.11 | CVPR2024    |







