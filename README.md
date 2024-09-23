# A Survey on conditional image synthesis with diffusion model

This repo is constructed for collecting and categorizing papers about conditional image synthesis with-diffusion-model.

## Paper Structure

![Conditional image synthesis with diffusion model](https://github.com/Szy12345-liv/A-Survey-on-conditional-image-synthesis-with-diffusion-model/blob/main/Images/Conditional%20image%20synthesis%20with%20diffusion%20model.png)


## Abstract

Conditional image synthesis based on user-specified requirements is a key component in creating complex visual content. In recent years, diffusion-based generative modeling has become a highly effective way for conditional image synthesis, leading to exponential growth in the literature. 
However, the complexity of diffusion-based modeling, the wide range of image synthesis tasks, and the diversity of conditioning mechanisms present significant challenges for researchers to keep up with rapid developments and understand the core concepts on this topic. 
In this survey, we categorize existing works based on how conditions are integrated into the two fundamental components of diffusion-based modeling, the denoising network and the sampling process. We specifically highlight the underlying principles, advantages, and potential challenges of various conditioning approaches in the training, re-purposing, and specialization stages to construct a desired denoising network. We also summarize six conditioning mechanisms in the essential sampling process. All discussions are centered around popular applications. Finally, we pinpoint some critical yet still open problems to be solved in the future and suggest some possible solutions.

## Contents
- [Condition Integration in Denoising Networks](#condition-integration-in-denoising-networks)
  - [Condition Integration in the Training Stage](#Condition Integration in the Training Stage)
    - [Conditional models for text-to-image (T2I)](#Conditional models for text-to-image (T2I))
    - [Conditional Models for Image Restoration](#Conditional Models for Image Restoration)
    - [Conditional Models for Other Synthesis Scenarios](#Conditional Models for Other Synthesis Scenarios)
  - [Condition Integration in the Re-purposing Stage](#Condition Integration in the Re-purposing Stage)
    - [Re-purposed Conditional Encoders](#Re-purposed Conditional Encoders)
    - [Condition Injection](#Condition Injection)
    - [Backbone Fine-tuning](#Backbone Fine-tuning)
  - [Condition Integration in the Specialization Stage](#Condition Integration in the Specialization Stage)
    - [Condition Injection](#Condition Injection)
    - [Testing-time Model Fine-Tuning](#Testing-time Model Fine-Tuning)
- [Condition Integration in the Sampling Process](#Condition Integration in the Sampling Process)
  - [Inversion](#Inversion)
  - [Attention Manipulation](#Attention Manipulation)
  - [Noise Blending](#Noise Blending)
  - [Revising Diffusion Process](#Revising Diffusion Process)   
  - [Guidance](#Guidance)
  - [Conditional Correction](#Conditional Correction)


# Papers

The date in the table represents the publication date of the first version of the paper on Arxiv.

## Condition Integration in Denoising Networks

### Condition Integration in the Training Stage

#### Conditional models for text-to-image (T2I)

| Title                                                        | Date    | Publication |
| ------------------------------------------------------------ | ------- | ----------- |
| [**High-resolution image synthesis with latent diffusion models**](https://arxiv.org/abs/2112.10752) | 2021.12 | CVPR2022    |
| [**Photorealistic text-to-image diffusion models with deep language understanding**](https://arxiv.org/abs/2205.11487) | 2022.5  | NeurIPS2022 |
| [**ediffi: Text-toimage diffusion models with an ensemble of expert denoisers**](https://arxiv.org/abs/2211.01324) | 2022.11 | ARXIV2022   |
| [**Vector quantized diffusion model for text-toimage synthesis**](https://arxiv.org/abs/2111.14822) | 2021.11 | CVPR2022    |
| [**GLIDE: towards photorealistic image generation and editing with text-guided diffusion models**](https://arxiv.org/abs/2112.10741) | 2021.12 | ICML2022    |
| [**Hierarchical text-conditional image generation with CLIP latents**](https://arxiv.org/abs/2204.06125) | 2022.4  | ARXIV2022   |

#### Conditional Models for Image Restoration

| Title                                                        | Date    | Publication        |
| ------------------------------------------------------------ | ------- | ------------------ |
| [**Image super-resolution via iterative refinement**](https://arxiv.org/abs/2104.07636) | 2021.4  | TPAMI2022          |
| [**Cascaded diffusion models for high fidelity image generation**](https://arxiv.org/abs/2106.15282) | 2021.5  | JMLR2022           |
| [**Palette: Image-to-image diffusion models**](https://arxiv.org/abs/2111.05826) | 2021.11 | SIGGRAPH2022       |
| [**Low-light image enhancement with wavelet-based diffusion models**](https://arxiv.org/abs/2306.00306) | 2023.6  | TOG2023            |
| [**Srdiff: Single image super-resolution with diffusion probabilistic models**](https://arxiv.org/abs/2104.14951) | 2021.4  | Neurocomputing2022 |
| [**Denoising diffusion probabilistic models for robust image super-resolution in the wild**](https://arxiv.org/abs/2302.07864) | 2023.2  | ARXIV2023          |
| [**Resdiff: Combining cnn and diffusion model for image super-resolution**](https://arxiv.org/abs/2303.08714) | 2023.3  | AAAI2024           |
| [**Low-light image enhancement via clip-fourier guided wavelet diffusion**](https://arxiv.org/abs/2401.03788) | 2024.1  | ARXIV2024          |
| [**Diffusion-based blind text image super-resolution**](https://arxiv.org/abs/2312.08886) | 2023.12 | CVPR2024           |
| [**Wavelet-based fourier information interaction with frequency diffusion adjustment for underwater image restoration**](https://arxiv.org/abs/2311.16845) | 2023.11 | CVPR2024           |

#### Conditional Models for Other Synthesis Scenarios

| Title                                                        | Date    | Publication              |
| ------------------------------------------------------------ | ------- | ------------------------ |
| [**Learned representation-guided diffusion models for large-image generation**](https://arxiv.org/abs/2312.07330) | 2023.12 | ARXIV2023                |
| [**Zero-shot medical image translation via frequency-guided diffusion models**](https://arxiv.org/abs/2304.02742) | 2023.4  | Trans. Med. Imaging 2023 |
| [**Dolce: A model-based probabilistic diffusion framework for limited-angle ct reconstruction**](https://arxiv.org/abs/2211.12340) | 2022.11 | ICCV2023                 |
| [**A novel unified conditional scorebased generative framework for multi-modal medical image completion**](https://arxiv.org/abs/2207.03430) | 2022.7  | ARXIV2022                |
| [**A morphology focused diffusion probabilistic model for synthesis of histopathology images**](https://arxiv.org/abs/2209.13167) | 2022.9  | WACV2023                 |
| [**Diffusion autoencoders: Toward a meaningful and decodable representation**](https://arxiv.org/abs/2111.15640) | 2021.11 | CVPR2022                 |
| [**Semantic image synthesis via diffusion models**](https://arxiv.org/abs/2207.00050) | 2022.6  | ARXIV2022                |
| [**Diffusion-based scene graph to image generation with masked contrastive pre-training**](https://arxiv.org/abs/2211.11138) | 2022.11 | ARXIV2022                |
| [**Humandiffusion: a coarse-to-fine alignment diffusion framework for controllable text-driven person image generation**](https://arxiv.org/abs/2211.06235) | 2022.11 | ARXIV2022                |

### Condition Integration in the Re-purposing Stage

#### Re-purposed Conditional Encoders

| Title                                                        | Date    | Publication  |
| ------------------------------------------------------------ | ------- | ------------ |
| [**T2i-adapter: Learning adapters to dig out more controllable ability for text-to-image diffusion models**](https://arxiv.org/abs/2302.08453) | 2023.2  | AAAI2024     |
| [**Adding conditional control to text-to-image diffusion models**](https://arxiv.org/abs/2302.05543) | 2023.2  | ICCV2023     |
| [**Pretraining is all you need for image-to-image translation**](https://arxiv.org/abs/2205.12952) | 2022.5  | ARXIV2022    |
| [**Blip-diffusion: pre-trained subject representation for controllable text-to-image generation and editing**](https://arxiv.org/abs/2305.14720) | 2023.5  | NeurIPS2023  |
| [**Guiding instruction-based image editing via multimodal large language models**](https://arxiv.org/abs/2309.17102) | 2023.9  | ARXIV2023    |
| [**Ranni: Taming text-to-image diffusion for accurate instruction following**](https://arxiv.org/abs/2311.17002) | 2023.11 | ARXIV2023    |
| [**Encoder-based domain tuning for fast personalization of text-to-image models**](https://arxiv.org/abs/2302.12228) | 2023.2  | TOG2023      |
| [**Pair-diffusion: Object-level image editing with structure-and-appearance paired diffusion models**](https://arxiv.org/abs/2303.17546v1) | 2023.3  | ARXIV2023    |
| [**Smartedit: Exploring complex instruction-based image editing with multimodal large language models**](https://arxiv.org/abs/2312.06739) | 2023.12 | ARXIV2023    |
| [**Taming encoder for zero fine-tuning image customization with text-to-image diffusion models**](https://arxiv.org/abs/2304.02642) | 2023.4  | ARXIV2023    |
| [**Lightit: Illumination modeling and control for diffusion models**](https://arxiv.org/abs/2403.10615) | 2024.3  | CVPR2024     |
| [**Instructany2pix: Flexible visual editing via multimodal instruction following**](https://arxiv.org/abs/2312.06738) | 2023.12 | ARXIV2023    |
| [**Warpdiffusion: Efficient diffusion model for high-fidelity virtual try-on**](https://arxiv.org/abs/2312.03667) | 2023.12 | ARXIV2023    |
| [**Coarse-to-fine latent diffusion for pose-guided person image synthesis**](https://arxiv.org/abs/2402.18078) | 2024.2  | CVPR2024     |
| [**Subject-diffusion: Open domain personalized text-to-image generation without test-time fine-tuning**](https://arxiv.org/abs/2307.11410) | 2023.7  | SIGGRAPH2024 |
| [**Instantbooth: Personalized text-to-image generation without test-time finetuning**](https://arxiv.org/abs/2304.03411) | 2023.4  | CVPR2024     |
| [**Face2diffusion for fast and editable face personalization**](https://arxiv.org/abs/2403.05094) | 2024.3  | CVPR2024     |
| [**Fastcomposer: Tuning-free multi-subject image generation with localized attention**](https://arxiv.org/abs/2305.10431) | 2023.5  | ARXIV2023    |
| [**Prompt-free diffusion: Taking” text” out of text-to-image diffusion models**](https://arxiv.org/abs/2305.16223) | 2023.5  | CVPR2024     |
| [**Imagebrush: Learning visual in-context instructions for exemplar-based image manipulation**](https://arxiv.org/abs/2308.00906) | 2023.8  | NeurIPS2024  |
| [**Paste,inpaint and harmonize via denoising: Subject-driven image editing with pre-trained diffusion model**](https://arxiv.org/abs/2306.07596) | 2023.6  | ARXIV2023    |

#### Condition Injection

| Title                                                        | Date    | Publication |
| ------------------------------------------------------------ | ------- | ----------- |
| [**Ip-adapter: Text compatible image prompt adapter for text-to-image diffusion models**](https://arxiv.org/abs/2308.06721) | 2023.8  | ARXIV2023   |
| [**GLIGEN: open-set grounded text-to-image generation**](https://arxiv.org/abs/2301.07093) | 2023.1  | CVPR2023    |
| [**Dragondiffusion: Enabling drag-style manipulation on diffusion models**](https://arxiv.org/abs/2307.02421) | 2023.7  | ICLR2024    |
| [**Mix-of-show: Decentralized low-rank adaptation for multi-concept customization of diffusion models**](https://arxiv.org/abs/2305.18292) | 2023.5  | NeurIPS2024 |
| [**Interactdiffusion: Interaction control in text-to-image diffusion models**](https://arxiv.org/abs/2312.05849) | 2023.12 | ARXIV2023   |
| [**Deadiff: An efficient stylization diffusion model with disentangled representations**](https://arxiv.org/abs/2403.06951) | 2024.3  | CVPR2024    |
| [**Instancediffusion: Instance-level control for image generation**](https://arxiv.org/abs/2402.03290) | 2024.2  | CVPR2024    |
| [**Elite: Encoding visual concepts into textual embeddings for customized text-to-image generation**](https://arxiv.org/abs/2302.13848) | 2023.2  | CVPR2023    |

#### Backbone Fine-tuning

| Title                                                        | Date    | Publication |
| ------------------------------------------------------------ | ------- | ----------- |
| [**Instructpix2pix: Learning to follow image editing instructions**](https://arxiv.org/abs/2211.09800) | 2022.11 | CVPR2023    |
| [**Paint by example: Exemplar-based image editing with diffusion models**](https://arxiv.org/abs/2211.13227) | 2022.11 | CVPR2023    |
| [**Anydoor: Zero-shot object-level image customization**](https://arxiv.org/abs/2307.09481) | 2023.7  | CVPR2024    |
| [**Instructdiffusion: A generalist modeling interface for vision tasks**](https://arxiv.org/abs/2309.03895) | 2023.9  | ARXIV2023   |
| [**Reference-based image composition with sketch via structure-aware diffusion model**](https://arxiv.org/abs/2304.09748) | 2023.3  | ARXIV2023   |
| [**Emu edit: Precise image editing via recognition and generation tasks**](https://arxiv.org/abs/2311.10089) | 2023.11 | CVPR2024    |
| [**Objectstitch: Object compositing with diffusion model**](https://arxiv.org/abs/2212.00932v1) | 2022.12 | CVPR2023    |
| [**Imagen editor and editbench: Advancing and evaluating text-guided image inpainting**](https://arxiv.org/abs/2212.06909) | 2022.12 | CVPR2023    |
| [**Dialogpaint: A dialogbased image editing model**](https://arxiv.org/abs/2303.10073) | 2023.3  | ARXIV2023   |
| [**Smartbrush: Text and shape guided object inpainting with diffusion model**](https://arxiv.org/abs/2212.05034) | 2022.12 | CVPR2023    |
| [**Dreaminpainter: Text-guided subject-driven image inpainting with diffusion models**](https://arxiv.org/abs/2312.03771) | 2023.12 | ARXIV2023   |
| [**Inst-inpaint: Instructing to remove objects with diffusion models**](https://arxiv.org/abs/2304.03246) | 2023.4  | ARXIV2023   |
| [**Magicbrush: A manually annotated dataset for instruction-guided image editing**](https://arxiv.org/abs/2306.10012) | 2023.6  | NeurIPS2024 |
| [**Hive: Harnessing human feedback for instructional visual editing**](https://arxiv.org/abs/2303.09618) | 2023.3  | CVPR2024    |
| [**Text-to-image editing by image information removal**](https://arxiv.org/abs/2305.17489) | 2023.5  | WACV2024    |

### Condition Integration in the Specialization Stage

#### Conditional Projection

| Title                                                        | Date    | Publication |
| ------------------------------------------------------------ | ------- | ----------- |
| [**Imagic: Text-based real image editing with diffusion models**](https://arxiv.org/abs/2210.09276) | 2022.10 | CVPR2023    |
| [**An image is worth one word: Personalizing text-to-image generation using textual inversion**](https://arxiv.org/abs/2208.01618) | 2022.8  | ICLR2023    |
| [**iedit: Localised text-guided image editing with weak supervision**](https://arxiv.org/abs/2305.05947) | 2023.5  | CVPR2024    |
| [**Prompting hard or hardly prompting: Prompt inversion for text-to-image diffusion models**](https://arxiv.org/abs/2312.12416) | 2023.12 | CVPR2024    |
| [**Preditor: Text guided image editing with diffusion prior**](https://arxiv.org/abs/2302.07979) | 2023.2  | ARXIV2023   |
| [**Uncovering the disentanglement capability in text-to-image diffusion models**](https://arxiv.org/abs/2212.08698) | 2022.12 | CVPR2023    |
| [**Forgedit: Text guided image editing via learning and forgetting**](https://arxiv.org/abs/2309.10556) | 2023.9  | ARXIV2023   |

#### Testing-time Model Fine-Tuning

| Title                                                        | Date    | Publication       |
| ------------------------------------------------------------ | ------- | ----------------- |
| [**Imagic: Text-based real image editing with diffusion models**](https://arxiv.org/abs/2210.09276) | 2022.10 | CVPR2023          |
| [**Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation**](https://arxiv.org/abs/2208.12242) | 2022.8  | CVPR2023          |
| [**Custom-edit: Text-guided image editing with customized diffusion models**](https://arxiv.org/abs/2305.15779) | 2023.5  | ARXIV2023         |
| [**Encoder-based domain tuning for fast personalization of text-to-image models**](https://arxiv.org/abs/2302.12228) | 2023.2  | TOG2023           |
| [**Mix-of-show: Decentralized low-rank adaptation for multi-concept customization of diffusion models**](https://arxiv.org/abs/2305.18292) | 2023.5  | NeurIPS2024       |
| [**Svdiff: Compact parameter space for diffusion fine-tuning**](https://arxiv.org/abs/2303.11305) | 2023.3  | ICCV2023          |
| [**Multi-concept customization of text-to-image diffusion**](https://arxiv.org/abs/2212.04488) | 2022.12 | CVPR2023          |
| [**Layerdiffusion: Layered controlled image editing with diffusion models**](https://arxiv.org/abs/2305.18676) | 2023.5  | SIGGRAPH Asia2023 |
| [**Cones: concept neurons in diffusion models for customized generation**](https://arxiv.org/abs/2303.05125) | 2023.3  | ICML2023          |
| [**Cones 2: Customizable image synthesis with multiple subjects**](https://arxiv.org/abs/2305.19327) | 2023.5  | NeurIPS2023       |
| [**Unitune: Text-driven image editing by fine tuning a diffusion model on a single image**](https://arxiv.org/abs/2210.09477) | 2022.10 | TOG2023           |
| [**Sine: Single image editing with text-to-image diffusion models**](https://arxiv.org/abs/2212.04489) | 2022.12 | CVPR2023          |

## Condition Integration in the Sampling Process

### Inversion

| Title                                                        | Date    | Publication |
| ------------------------------------------------------------ | ------- | ----------- |
| [**Sdedit: Guided image synthesis and editing with stochastic differential equations**](https://arxiv.org/abs/2108.01073) | 2021.8  | ICLR2022    |
| [**Dual diffusion implicit bridges for image-to-image translation**](https://arxiv.org/abs/2203.08382) | 2022.3  | ICLR2023    |
| [**Null-text inversion for editing real images using guided diffusion models**](https://arxiv.org/abs/2211.09794) | 2022.11 | CVPR2023    |
| [**A latent space of stochastic diffusion models for zero-shot image editing and guidance**](https://openaccess.thecvf.com/content/ICCV2023/papers/Wu_A_Latent_Space_of_Stochastic_Diffusion_Models_for_Zero-Shot_Image_ICCV_2023_paper.pdf) |         | ICCV2023    |
| [**An edit friendly ddpm noise space: Inversion and manipulations**](https://arxiv.org/abs/2304.06140) | 2023.4  | ARXIV2023   |
| [**Ledits++: Limitless image editing using text-to-image models**](https://arxiv.org/abs/2311.16711) | 2023.11 | CVPR2024    |
| [**Style injection in diffusion: A training-free approach for adapting large-scale diffusion models for style transfer**](https://arxiv.org/abs/2312.09008) | 2023.12 | CVPR2024    |
| [**Prompt tuning inversion for text-driven image editing using diffusion models**](https://arxiv.org/abs/2305.04441) | 2023.5  | ICCV2023    |
| [**Kv inversion: Kv embeddings learning for text-conditioned real image action editing**](https://arxiv.org/abs/2309.16608) | 2023.9  | PRCV2023    |
| [**Direct inversion: Boosting diffusion-based editing with 3 lines of code**](https://arxiv.org/abs/2310.01506) | 2023.10 | ARXIV2023   |
| [**Tf-icon: Diffusion-based training-free cross-domain image composition**](https://arxiv.org/abs/2307.12493) | 2023.7  | ICCV2023    |
| [**Fixed-point inversion for text-to-image diffusion models**](https://arxiv.org/abs/2312.12540v1) | 2023.12 | ARXIV2023   |
| [**Negative-prompt inversion: Fast image inversion for editing with textguided diffusion models**](https://arxiv.org/abs/2305.16807) | 2023.5  | ARXIV2023   |
| [**The blessing of randomness: Sde beats ode in general diffusionbased image editing**](https://arxiv.org/abs/2311.01410) | 2023.11 | ICLR2023    |
| [**Effective real image editing with accelerated iterative diffusion inversion**](https://arxiv.org/abs/2309.04907) | 2023.9  | ICCV2023    |
| [**Dragdiffusion: Harnessing diffusion models for interactive point-based image editing**](https://arxiv.org/abs/2306.14435) | 2023.6  | CVPR2024    |
| [**Edict: Exact diffusion inversion via coupled transformations**](https://arxiv.org/abs/2211.12446) | 2022.11 | CVPR2023    |
| [**Stylediffusion: Controllable disentangled style transfer via diffusion models**](https://arxiv.org/abs/2308.07863) | 2023.8  | ICCV2023    |
| [**Inversion-based style transfer with diffusion models**](https://arxiv.org/abs/2211.13203) | 2022.11 | CVPR2023    |

### Attention Manipulation

| Title                                                        | Date    | Publication |
| ------------------------------------------------------------ | ------- | ----------- |
| [**Prompt-to-prompt image editing with cross attention control**](https://arxiv.org/abs/2208.01626) | 2022.8  | ICLR2023    |
| [**Plug-and-play diffusion features for text-driven image-to-image translation**](https://arxiv.org/abs/2211.12572) | 2022.11 | CVPR2023    |
| [**Masactrl: Tuning-free mutual self-attention control for consistent image synthesis and editing**](https://arxiv.org/abs/2304.08465) | 2023.4  | ICCV2023    |
| [**ediffi: Text-toimage diffusion models with an ensemble of expert denoisers**](https://arxiv.org/abs/2211.01324) | 2022.11 | ARXIV2022   |
| [**Face aging via diffusion-based editing**](https://arxiv.org/abs/2309.11321) | 2023.9  | BMVC2023    |
| [**Custom-edit: Text-guided image editing with customized diffusion models**](https://arxiv.org/abs/2305.15779) | 2023.5  | ARXIV2023   |
| [**Style injection in diffusion: A training-free approach for adapting large-scale diffusion models for style transfer**](https://arxiv.org/abs/2312.09008) | 2023.12 | CVPR2024    |
| [**Focus on your instruction: Fine-grained and multi-instruction image editing by attention modulation**](https://arxiv.org/abs/2312.10113) | 2023.12 | ARXIV2023   |
| [**Towards understanding cross and self-attention in stable diffusion for text-guided image editing**](https://arxiv.org/abs/2403.03431) | 2024.3  | CVPR2024    |
| [**Cones 2: Customizable image synthesis with multiple subjects**](https://arxiv.org/abs/2305.19327) | 2023.5  | NeurIPS2023 |
| [**Tf-icon: Diffusion-based training-free cross-domain image composition**](https://arxiv.org/abs/2307.12493) | 2023.7  | ICCV2023    |
| [**Dragondiffusion: Enabling drag-style manipulation on diffusion models**](https://arxiv.org/abs/2307.02421) | 2023.7  | ICLR2024    |
| [**Dragdiffusion: Harnessing diffusion models for interactive point-based image editing**](https://arxiv.org/abs/2306.14435) | 2023.6  | CVPR2024    |
| [**Stylediffusion: Controllable disentangled style transfer via diffusion models**](https://arxiv.org/abs/2308.07863) | 2023.8  | ICCV2023    |
| [**Dynamic prompt learning: Addressing cross-attention leakage for text-based image editing**](https://arxiv.org/abs/2309.15664) | 2023.9  | NeurIPS2024 |

### Noise Blending

| Title                                                        | Date    | Publication |
| ------------------------------------------------------------ | ------- | ----------- |
| [**Compositional visual generation with composable diffusion models**](https://arxiv.org/abs/2206.01714) | 2022.6  | ECCV2022    |
| [**Classifier-free diffusion guidance**](https://arxiv.org/abs/2207.12598) | 2022.7  | ARXIV2022   |
| [**Multidiffusion: Fusing diffusion paths for controlled image generation**](https://arxiv.org/abs/2302.08113) | 2023.2  | ICML2023    |
| [**Ledits++: Limitless image editing using text-to-image models**](https://arxiv.org/abs/2311.16711) | 2023.11 | CVPR2024    |
| [**Pair-diffusion: Object-level image editing with structure-and-appearance paired diffusion models**](https://arxiv.org/abs/2303.17546) | 2023.3  | ARXIV2023   |
| [**Effective real image editing with accelerated iterative diffusion inversion**](https://arxiv.org/abs/2309.04907) | 2023.9  | ICCV2023    |
| [**Noisecollage: A layout-aware text-to-image diffusion model based on noise cropping and merging**](https://arxiv.org/abs/2403.03485) | 2024.3  | CVPR2024    |
| [**Sine: Single image editing with text-to-image diffusion models**](https://arxiv.org/abs/2212.04489) | 2022.12 | CVPR2023    |
| [**Magicfusion: Boosting text-to-image generation performance by fusing diffusion models**](https://arxiv.org/abs/2303.13126) | 2023.3  | ICCV2023    |

### Revising Diffusion Process

| Title                                                        | Date    | Publication |
| ------------------------------------------------------------ | ------- | ----------- |
| [**Image restoration with mean-reverting stochastic differential equations**](https://arxiv.org/abs/2301.11699) | 2023.1  | ICML2023    |
| [**Snips: Solving noisy inverse problems stochastically**](https://arxiv.org/abs/2105.14951) | 2021.5  | NeurIPS2021 |
| [**Denoising diffusion restoration models**](https://arxiv.org/abs/2201.11793) | 2022.1  | NeurIPS2022 |
| [**Inversion by direct iteration: An alternative to denoising diffusion for image restoration**](https://arxiv.org/abs/2303.11435) | 2023.3  | TMLR2023    |
| [**Sinsr: diffusion-based image super-resolution in a single step. **](https://arxiv.org/abs/2311.14760) | 2023.11 | CVPR2024    |
| [**Zero-shot image restoration using denoising diffusion null-space model**](https://arxiv.org/abs/2212.00490) | 2022.12 | ICLR2024    |
| [**Driftrec: Adapting diffusion models to blind jpeg restoration**](https://arxiv.org/abs/2211.06757) | 2022.11 | TIP2024     |
| [**Resshift: Efficient diffusion model for image super-resolution by residual shifting**](https://arxiv.org/abs/2307.12348) | 2023.7  | NeurIPS2024 |

### Guidance

| Title                                                        | Date    | Publication  |
| ------------------------------------------------------------ | ------- | ------------ |
| [**Diffusion models beat gans on image synthesis**](https://arxiv.org/abs/2105.05233) | 2021.5  | NeurIPS2021  |
| [**Improving diffusion models for inverse problems using manifold constraints**](https://arxiv.org/abs/2206.00941) | 2022.6  | NeurIPS2022  |
| [**Diffusion posterior sampling for general noisy inverse problems**](https://arxiv.org/abs/2209.14687) | 2022.9  | ICLR2023     |
| [**Blended diffusion for text-driven editing of natural images**](https://arxiv.org/abs/2111.14818) | 2021.11 | CVPR2022     |
| [**Sketch-guided text-to-image diffusion models**](https://arxiv.org/abs/2211.13752) | 2022.11 | SIGGRAPH2023 |
| [**Zero-shot image-to-image translation**](https://arxiv.org/abs/2302.03027) | 2023.2  | SIGGRAPH2023 |
| [**Freedom: Training-free energy-guided conditional diffusion model**](https://arxiv.org/abs/2303.09833) | 2023.3  | ICCV2023     |
| [**Universal guidance for diffusion models**](https://arxiv.org/abs/2302.07121) | 2023.2  | CVPR2023     |
| [**Training-free layout control with cross-attention guidance**](https://arxiv.org/abs/2304.03373) | 2023.4  | WACV2024     |
| [**Parallel diffusion models of operator and image for blind inverse problems**](https://arxiv.org/abs/2211.10656) | 2022.11 | CVPR2023     |
| [**Diffusion self-guidance for controllable image generation**](https://arxiv.org/abs/2306.00986) | 2023.6  | NeurIPS2024  |
| [**Generative diffusion prior for unified image restoration and enhancement**](https://arxiv.org/abs/2304.01247) | 2023.4  | CVPR2023     |
| [**Diffusion-based image translation using disentangled style and content representation**](https://arxiv.org/abs/2209.15264) | 2022.9  | ICLR2023     |
| [**Regeneration learning of diffusion models with rich prompts for zero-shot image translation**](https://arxiv.org/abs/2305.04651) | 2023.5  | ARXIV2023    |
| [**More control for free! image synthesis with semantic diffusion guidance**](https://arxiv.org/abs/2112.05744) | 2021.12 | WACV2023     |
| [**Readout guidance: Learning control from diffusion features**](https://arxiv.org/abs/2312.02150) | 2023.12 | CVPR2024     |
| [**Freecontrol: Training-free spatial control of any text-to-image diffusion model with any condition**](https://arxiv.org/abs/2312.07536) | 2023.12 | CVPR2024     |
| [**Diffeditor: Boosting accuracy and flexibility on diffusion-based image editing**](https://arxiv.org/abs/2402.02583) | 2024.2  | CVPR2024     |
| [**Dragondiffusion: Enabling drag-style manipulation on diffusion models**](https://arxiv.org/abs/2307.02421) | 2023.7  | ICLR2024     |
| [**Energy-based cross attention for bayesian context update in text-to-image diffusion models**](https://arxiv.org/abs/2306.09869) | 2023.6  | NeurIPS2024  |
| [**Solving linear inverse problems provably via posterior sampling with latent diffusion models**](https://arxiv.org/abs/2307.00619) | 2023.7  | NeurIPS2024  |
| [**High-fidelity guided image synthesis with latent diffusion models**](https://arxiv.org/abs/2211.17084) | 2022.11 | CVPR2023     |
| [**Pseudoinverse-guided diffusion models for inverse problems **](https://openreview.net/pdf?id=9_gsMA8MRKQ) | 2023.2  | ICLR2023     |
| [**Freedom: Training-free energy-guided conditional diffusion model**](https://arxiv.org/abs/2303.09833) | 2023.3  | ICCV2023     |

### Conditional Correction

| Title                                                        | Date    | Publication |
| ------------------------------------------------------------ | ------- | ----------- |
| [**Score-based generative modeling through stochastic differential equations**](https://arxiv.org/abs/2011.13456) | 2020.11 | ICLR2021    |
| [**Repaint: Inpainting using denoising diffusion probabilistic models**](https://arxiv.org/abs/2201.09865) | 2022.1  | CVPR2022    |
| [**ILVR: conditioning method for denoising diffusion probabilistic models**](https://arxiv.org/abs/2108.02938) | 2021.8  | ICCV2021    |
| [**Diffedit: Diffusion-based semantic image editing with mask guidance**](https://arxiv.org/abs/2210.11427) | 2022.10 | ICLR2023    |
| [**Come-closer-diffuse-faster: Accelerating conditional diffusion models for inverse problems through stochastic contraction**](https://arxiv.org/abs/2112.05146) | 2021.12 | CVPR2022    |
| [**Improving diffusion models for inverse problems using manifold constraints.**](https://arxiv.org/abs/2206.00941) | 2022.6  | NeurIPS2022 |
| [**Region-aware diffusion for zero-shot text-driven image editing**](https://arxiv.org/abs/2302.11797) | 2023.2  | ARXIV2023   |
| [**Text-driven image editing via learnable regions**](https://arxiv.org/abs/2311.16432) | 2023.11 | CVPR2024    |
| [**Localizing object-level shape variations with text-to-image diffusion models**](https://arxiv.org/abs/2303.11306) | 2023.3  | ICCV2023    |
| [**Instructedit: Improving automatic masks for diffusion-based image editing with user instructions**](https://arxiv.org/abs/2305.18047) | 2023.5  | ARXIV2023   |





