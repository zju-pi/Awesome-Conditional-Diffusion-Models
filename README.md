# A Survey on conditional image synthesis with diffusion model

This repo is constructed for collecting and categorizing papers about conditional image synthesis with-diffusion-model.

## Paper Structure

![Conditional image synthesis with diffusion model](https://github.com/Szy12345-liv/A-Survey-on-conditional-image-synthesis-with-diffusion-model/blob/main/Images/Conditional%20image%20synthesis%20with%20diffusion%20model.png)

## Abstract

Conditional image synthesis based on user-specified requirements is a key component in creating complex visual content. In recent years, diffusion-based generative modeling has become a highly effective way for conditional image synthesis, leading to exponential growth in the literature. 
However, the complexity of diffusion-based modeling, the wide range of image synthesis tasks, and the diversity of conditioning mechanisms present significant challenges for researchers to keep up with rapid developments and understand the core concepts on this topic. 
In this survey, we categorize existing works based on how conditions are integrated into the two fundamental components of diffusion-based modeling, the denoising network and the sampling process. We specifically highlight the underlying principles, advantages, and potential challenges of various conditioning approaches in the training, re-purposing, and specialization stages to construct a desired denoising network. We also summarize six conditioning mechanisms in the essential sampling process. All discussions are centered around popular applications. Finally, we pinpoint some critical yet still open problems to be solved in the future and suggest some possible solutions.

**The date in the table represents the publication date of the first version of the paper on Arxiv.**

## Contents

[TOC]



# Papers

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

| Title                                                        | Date | Publication        |
| ------------------------------------------------------------ | ---- | ------------------ |
| **Image super-resolution via iterative refinement**          |      | TPAMI2022          |
| **Cascaded diffusion models for high fidelity image generation** |      | JMLR2022           |
| **Palette: Image-to-image diffusion models**                 |      | SIGGRAPH2022       |
| **Low-light image enhancement with wavelet-based diffusion models** |      | TOG2023            |
| **Srdiff: Single image super-resolution with diffusion probabilistic models** |      | Neurocomputing2022 |
| **Denoising diffusion probabilistic models for robust image super-resolution in the wild** |      | ARXIV2023          |
| **Resdiff: Combining cnn and diffusion model for image super-resolution** |      | AAAI2024           |
| **Low-light image enhancement via clip-fourier guided wavelet diffusion** |      | ARXIV2024          |
| **Diffusion-based blind text image super-resolution**        |      | CVPR2024           |
| **Wavelet-based fourier information interaction with frequency diffusion adjustment for underwater image restoration** |      | CVPR2024           |

#### Conditional Models for Other Synthesis Scenarios

| Title                                                        | Date | Publication              |
| ------------------------------------------------------------ | ---- | ------------------------ |
| **Learned representation-guided diffusion models for large-image generation** |      | ARXIV2023                |
| **Zero-shot medical image translation via frequency-guided diffusion models** |      | Trans. Med. Imaging 2023 |
| **Dolce: A model-based probabilistic diffusion framework for limited-angle ct reconstruction** |      | ICCV2023                 |
| **A novel unified conditional scorebased generative framework for multi-modal medical image completion** |      | ARXIV2022                |
| **A morphology focused diffusion probabilistic model for synthesis of histopathology images** |      | WACV2023                 |
| **Diffusion autoencoders: Toward a meaningful and decodable representation** |      | CVPR2022                 |
| **Semantic image synthesis via diffusion models**            |      | ARXIV2022                |
| **Diffusion-based scene graph to image generation with masked contrastive pre-training** |      | ARXIV2022                |
| **Humandiffusion: a coarse-to-fine alignment diffusion framework for controllable text-driven person image generation** |      | ARXIV2022                |

### Condition Integration in the Re-purposing Stage

#### Re-purposed Conditional Encoders

| Title                                                        | Date | Publication  |
| ------------------------------------------------------------ | ---- | ------------ |
| **T2i-adapter: Learning adapters to dig out more controllable ability for text-to-image diffusion models** |      | AAAI2024     |
| **Adding conditional control to text-to-image diffusion models** |      | ICCV2023     |
| **Pretraining is all you need for image-to-image translation** |      | ARXIV2022    |
| **Blip-diffusion: pre-trained subject representation for controllable text-to-image generation and editing** |      | NeurIPS2023  |
| **Guiding instruction-based image editing via multimodal large language models** |      | ARXIV2023    |
| **Ranni: Taming text-to-image diffusion for accurate instruction following** |      | ARXIV2023    |
| **Encoder-based domain tuning for fast personalization of text-to-image models** |      | TOG2023      |
| **Pair-diffusion: Object-level image editing with structure-and-appearance paired diffusion models** |      | ARXIV2023    |
| **Smartedit: Exploring complex instruction-based image editing with multimodal large language models** |      | ARXIV2023    |
| **Taming encoder for zero fine-tuning image customization with text-to-image diffusion models** |      | ARXIV2023    |
| **Lightit: Illumination modeling and control for diffusion models** |      | CVPR2024     |
| **Instructany2pix: Flexible visual editing via multimodal instruction following** |      | ARXIV2023    |
| **Warpdiffusion: Efficient diffusion model for high-fidelity virtual try-on** |      | ARXIV2023    |
| **Coarse-to-fine latent diffusion for pose-guided person image synthesis** |      | CVPR2024     |
| **Subject-diffusion: Open domain personalized text-to-image generation without test-time fine-tuning** |      | SIGGRAPH2024 |
| **Instantbooth: Personalized text-to-image generation without test-time finetuning** |      | CVPR2024     |
| **Face2diffusion for fast and editable face personalization** |      | CVPR2024     |
| **Fastcomposer: Tuning-free multi-subject image generation with localized attention** |      | ARXIV2023    |
| **Prompt-free diffusion: Taking” text” out of text-to-image diffusion models** |      | CVPR2024     |
| **Imagebrush: Learning visual in-context instructions for exemplar-based image manipulation** |      | NeurIPS2024  |
| **inpaint and harmonize via denoising: Subject-driven image editing with pre-trained diffusion model** |      | ARXIV2023    |

#### Condition Injection

| Title                                                        | Date | Publication |
| ------------------------------------------------------------ | ---- | ----------- |
| **Ip-adapter: Text compatible image prompt adapter for text-to-image diffusion models** |      | ARXIV2023   |
| **GLIGEN: open-set grounded text-to-image generation**       |      | CVPR2023    |
| **Dragondiffusion: Enabling drag-style manipulation on diffusion models** |      | ICLR2024    |
| **Mix-of-show: Decentralized low-rank adaptation for multi-concept customization of diffusion models** |      | NeurIPS2024 |
| **Interactdiffusion: Interaction control in text-to-image diffusion models** |      | ARXIV2023   |
| **Deadiff: An efficient stylization diffusion model with disentangled representations** |      | CVPR2024    |
| **Instancediffusion: Instance-level control for image generation** |      | CVPR2024    |
| **Elite: Encoding visual concepts into textual embeddings for customized text-to-image generation** |      | CVPR2023    |

#### Backbone Fine-tuning

| Title                                                        | Date | Publication |
| ------------------------------------------------------------ | ---- | ----------- |
| **Instructpix2pix: Learning to follow image editing instructions** |      | CVPR2023    |
| **Paint by example: Exemplar-based image editing with diffusion models** |      | CVPR2023    |
| **Anydoor: Zero-shot object-level image customization**      |      | CVPR2024    |
| **Instructdiffusion: A generalist modeling interface for vision tasks** |      | ARXIV2023   |
| **Reference-based image composition with sketch via structure-aware diffusion model** |      | ARXIV2023   |
| **Emu edit: Precise image editing via recognition and generation tasks** |      | CVPR2024    |
| **Objectstitch: Object compositing with diffusion model**    |      | CVPR2023    |
| **Imagen editor and editbench: Advancing and evaluating textguided image inpainting** |      | CVPR2023    |
| **Dialogpaint: A dialogbased image editing model**           |      | ARXIV2023   |
| **Smartbrush: Text and shape guided object inpainting with diffusion model** |      | CVPR2023    |
| **Dreaminpainter: Text-guided subject-driven image inpainting with diffusion models** |      | ARXIV2023   |
| **Inst-inpaint: Instructing to remove objects with diffusion models** |      | ARXIV2023   |
| **Magicbrush: A manually annotated dataset for instruction-guided image editing** |      | NeurIPS2024 |
| **Hive: Harnessing human feedback for instructional visual editing** |      | CVPR2024    |
| **Text-toimage editing by image information removal**        |      | WACV2024    |

### Condition Integration in the Specialization Stage

#### Conditional Projection

| Title                                                        | Date | Publication |
| ------------------------------------------------------------ | ---- | ----------- |
| **Imagic: Text-based real image editing with diffusion models** |      | CVPR2023    |
| **An image is worth one word: Personalizing text-to-image generation using textual inversion** |      | ICLR2023    |
| **iedit: Localised text-guided image editing with weak supervision** |      | CVPR2024    |
| **Prompting hard or hardly prompting: Prompt inversion for text-to-image diffusion models** |      | CVPR2024    |
| **Preditor: Text guided image editing with diffusion prior** |      | ARXIV2023   |
| **Uncovering the disentanglement capability in text-to-image diffusion models** |      | CVPR2023    |
| **Forgedit: Text guided image editing via learning and forgetting** |      | ARXIV2023   |

#### Testing-time Model Fine-Tuning

| Title                                                        | Date | Publication       |
| ------------------------------------------------------------ | ---- | ----------------- |
| **Imagic: Text-based real image editing with diffusion models** |      | CVPR2023          |
| **Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation** |      | CVPR2023          |
| **Custom-edit: Text-guided image editing with customized diffusion models** |      | ARXIV2023         |
| **Encoder-based domain tuning for fast personalization of text-to-image models** |      | TOG2023           |
| **Mix-of-show: Decentralized low-rank adaptation for multi-concept customization of diffusion models** |      | NeurIPS2024       |
| **Svdiff: Compact parameter space for diffusion fine-tuning** |      | ICCV2023          |
| **Multi-concept customization of text-to-image diffusion**   |      | CVPR2023          |
| **Layerdiffusion: Layered controlled image editing with diffusion models** |      | SIGGRAPH Asia2023 |
| **Cones: concept neurons in diffusion models for customized generation** |      | ICML2023          |
| **Cones 2: Customizable image synthesis with multiple subjects** |      | NeurIPS2023       |
| **Unitune: Text-driven image editing by fine tuning a diffusion model on a single image** |      | TOG2023           |
| **Sine: Single image editing with text-to-image diffusion models** |      | CVPR2023          |

## Condition Integration in the Sampling Process

### Inversion

| Title                                                        | Date | Publication |
| ------------------------------------------------------------ | ---- | ----------- |
| **Sdedit: Guided image synthesis and editing with stochastic differential equations** |      | ICLR2022    |
| **Dual diffusion implicit bridges for image-to-image translation** |      | ICLR2023    |
| **Null-text inversion for editing real images using guided diffusion models** |      | CVPR2023    |
| **A latent space of stochastic diffusion models for zero-shot image editing and guidance** |      | ICCV2023    |
| **An edit friendly ddpm noise space: Inversion and manipulations** |      | ARXIV2023   |
| **Ledits++: Limitless image editing using text-to-image models** |      | CVPR2024    |
| **Style injection in diffusion: A training-free approach for adapting large-scale diffusion models for style transfer** |      | CVPR2024    |
| **Prompt tuning inversion for text-driven image editing using diffusion models** |      | ICCV2023    |
| **Kv inversion: Kv embeddings learning for text-conditioned real image action editing** |      | PRCV2023    |
| **Direct inversion: Boosting diffusion-based editing with 3 lines of code** |      | ARXIV2023   |
| **Tf-icon: Diffusion-based training-free cross-domain image composition** |      | ICCV2023    |
| **Fixed-point inversion for text-to-image diffusion models** |      | ARXIV2023   |
| **Negativeprompt inversion: Fast image inversion for editing with textguided diffusion models** |      | ARXIV2023   |
| **The blessing of randomness: Sde beats ode in general diffusionbased image editing** |      | ICLR2023    |
| **Effective real image editing with accelerated iterative diffusion inversion** |      | ICCV2023    |
| **Dragdiffusion: Harnessing diffusion models for interactive point-based image editing** |      | CVPR2024    |
| **Edict: Exact diffusion inversion via coupled transformations** |      | CVPR2023    |
| **Stylediffusion: Controllable disentangled style transfer via diffusion models** |      | ICCV2023    |
| **Inversion-based style transfer with diffusion models**     |      | CVPR2023    |

### Attention Manipulation

| Title                                                        | Date | Publication |
| ------------------------------------------------------------ | ---- | ----------- |
| **Prompt-to-prompt image editing with crossattention control** |      | ICLR2023    |
| **Plug-andplay diffusion features for text-driven image-to-image translation** |      | CVPR2023    |
| **Masactrl: Tuning-free mutual self-attention control for consistent image synthesis and editing** |      | ICCV2023    |
| **ediffi: Text-toimage diffusion models with an ensemble of expert denoisers** |      | ARXIV2022   |
| **Face aging via diffusion-based editing**                   |      | BMVC2023    |
| **Custom-edit: Text-guided image editing with customized diffusion models** |      | ARXIV2023   |
| **Style injection in diffusion: A training-free approach for adapting large-scale diffusion models for style transfer** |      | CVPR2024    |
| **Focus on your instruction: Fine-grained and multi-instruction image editing by attention modulation** |      | ARXIV2023   |
| **Towards understanding cross and self-attention in stable diffusion for text-guided image editing** |      | CVPR2024    |
| **Cones 2: Customizable image synthesis with multiple subjects** |      | NeurIPS2023 |
| **Tf-icon: Diffusion-based training-free cross-domain image composition** |      | ICCV2023    |
| **Dragondiffusion: Enabling drag-style manipulation on diffusion models** |      | ICLR2024    |
| **Dragdiffusion: Harnessing diffusion models for interactive point-based image editing** |      | CVPR2024    |
| **Stylediffusion: Controllable disentangled style transfer via diffusion models** |      | ICCV2023    |
| **Dynamic prompt learning: Addressing cross-attention leakage for textbased image editing** |      | NeurIPS2024 |

### Noise Blending

| Title                                                        | Date | Publication |
| ------------------------------------------------------------ | ---- | ----------- |
| **Compositional visual generation with composable diffusion models** |      | ECCV2022    |
| **Classifier-free diffusion guidance**                       |      | ARXIV2022   |
| **Multidiffusion: Fusing diffusion paths for controlled image generation** |      | ICML2023    |
| **Ledits++: Limitless image editing using text-to-image models** |      | CVPR2024    |
| **Pair-diffusion: Object-level image editing with structure-and-appearance paired diffusion models** |      | ARXIV2023   |
| **Effective real image editing with accelerated iterative diffusion inversion** |      | ICCV2023    |
| **Noisecollage: A layout-aware text-to-image diffusion model based on noise cropping and merging** |      | CVPR2024    |
| **Sine: Single image editing with text-to-image diffusion models** |      | CVPR2023    |
| **Magicfusion: Boosting text-to-image generation performance by fusing diffusion models** |      | ICCV2023    |

### Revising Diffusion Process

| Title                                                        | Date | Publication |
| ------------------------------------------------------------ | ---- | ----------- |
| **Image restoration with mean-reverting stochastic differential equations** |      | ICML2023    |
| **Snips: Solving noisy inverse problems stochastically**     |      | NeurIPS2021 |
| **Denoising diffusion restoration models**                   |      | NeurIPS2022 |
| **M. Delbracio and P. Milanfar. Inversion by direct iteration: An alternative to denoising diffusion for image restoration** |      | TMLR2023    |
| **Sinsr: diffusion-based image super-resolution in a single step. ** |      | CVPR2024    |
| **Zero-shot image restoration using denoising diffusion null-space model** |      | ICLR2024    |
| **Driftrec: Adapting diffusion models to blind jpeg restoration** |      | TIP2024     |
| **Resshift: Efficient diffusion model for image super-resolution by residual shifting** |      | NeurIPS2024 |

### Guidance

| Title                                                        | Date | Publication  |
| ------------------------------------------------------------ | ---- | ------------ |
| **Diffusion models beat gans on image synthesis**            |      | NeurIPS2021  |
| **Improving diffusion models for inverse problems using manifold constraints** |      | NeurIPS2022  |
| **Diffusion posterior sampling for general noisy inverse problems** |      | ICLR2023     |
| **Blended diffusion for text-driven editing of natural images** |      | CVPR2022     |
| **Sketch-guided text-to-image diffusion models.**            |      | SIGGRAPH2023 |
| **Zero-shot image-to-image translation**                     |      | SIGGRAPH2023 |
| **Freedom: Training-free energy-guided conditional diffusion model** |      | ICCV2023     |
| **Universal guidance for diffusion models**                  |      | CVPR2023     |
| **Training-free layout control with cross-attention guidance** |      | WACV2024     |
| **Parallel diffusion models of operator and image for blind inverse problems** |      | CVPR2023     |
| **Diffusion self-guidance for controllable image generation** |      | NeurIPS2024  |
| **Generative diffusion prior for unified image restoration and enhancement** |      | CVPR2023     |
| **Diffusion-based image translation using disentangled style and content representation** |      | ICLR2023     |
| **Regeneration learning of diffusion models with rich prompts for zero-shot image translation** |      | ARXIV2023    |
| **More control for free! image synthesis with semantic diffusion guidance** |      | WACV2023     |
| **Readout guidance: Learning control from diffusion features** |      | CVPR2024     |
| **Freecontrol: Training-free spatial control of any text-to-image diffusion model with any condition** |      | CVPR2024     |
| **Diffeditor: Boosting accuracy and flexibility on diffusion-based image editing** |      | CVPR2024     |
| **Dragondiffusion: Enabling drag-style manipulation on diffusion models** |      | ICLR2024     |
| **Energybased cross attention for bayesian context update in text-to-image diffusion models** |      | NeurIPS2024  |
| **Solving linear inverse problems provably via posterior sampling with latent diffusion models.** |      | NeurIPS2024  |
| **High-fidelity guided image synthesis with latent diffusion models** |      | CVPR2023     |
| **Pseudoinverseguided diffusion models for inverse problems. ** |      | ICLR2023     |
| **Freedom: Training-free energy-guided conditional diffusion model** |      | ICCV2023     |

### Conditional Correction

| Title                                                        | Date | Publication |
| ------------------------------------------------------------ | ---- | ----------- |
| **Score-based generative modeling through stochastic differential equations** |      | ICLR2021    |
| **Repaint: Inpainting using denoising diffusion probabilistic models** |      | CVPR2022    |
| **ILVR: conditioning method for denoising diffusion probabilistic models** |      | ICCV2021    |
| **Diffedit: Diffusion-based semantic image editing with mask guidance** |      | ICLR2023    |
| **Come-closer-diffuse-faster: Accelerating conditional diffusion models for inverse problems through stochastic contraction** |      | CVPR2022    |
| **Improving diffusion models for inverse problems using manifold constraints.** |      | NeurIPS2022 |
| **Region-aware diffusion for zero-shot text-driven image editing** |      | ARXIV2023   |
| **Text-driven image editing via learnable regions**          |      | CVPR2024    |
| **Localizing object-level shape variations with text-to-image diffusion models** |      | ICCV2023    |
| **Instructedit: Improving automatic masks for diffusion-based image editing with user instructions** |      | ARXIV2023   |





