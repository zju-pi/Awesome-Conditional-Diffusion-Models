# A Survey on conditional image synthesis with diffusion model

This repo is constructed for collecting and categorizing papers about conditional image synthesis with-diffusion-model.

## A mind map

![Conditional image synthesis with diffusion model](https://github.com/Szy12345-liv/A-Survey-on-conditional-image-synthesis-with-diffusion-model/blob/main/Images/Conditional%20image%20synthesis%20with%20diffusion%20model.png)

## Contents
- [Domain Translation](#Domain Translation)
- [Text-to-image](#Text-to-image)
  - [Diffusion in Latent space](##Diffusion in Latent space)

- [Semantic maps to image](#Semantic maps to image)
  - [Text edit with CLIP](##Text edit with CLIP)
  - [Exploit big T2I models](##Exploit big T2I models)
  - [CFG Inversion Technics](##CFG Inversion Technics)

- [Image editing](#Image Editing)
- [Image restoration](#Image restoration)
- [Auxiliary methods for condition injecting](#Auxiliary methods for condition injecting)

# Domain Translation
| Paper                                                        | Model  | Arxiv                                                | Date       | Comments  | Code |
| ------------------------------------------------------------ | ------ | ---------------------------------------------------- | ---------- | --------- | ---- |
| ILVR: Conditioning Method for Denoising Diffusion Probabilistic Models | ILVR   | [arXiv:2108.02938](https://arxiv.org/abs/2108.02938) | 2021.08.06 | ICCV 2021 |      |
| SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations | SDEdit | [arXiv:2108.01073](https://arxiv.org/abs/2108.01073) | 2021.08.02 | ICLR 2022 |      |
| Dual Diffusion Implicit Bridges for Image-to-Image Translation | DDIB   | [arXiv:2203.08382](https://arxiv.org/abs/2203.08382) | 2022.03.04 | ICLR 2023 |      |



# Text-to-image

## Diffusion in Latent space

| Paper                                                        | Model            | Arxiv                                                | Date       | Comments  | Code |
| ------------------------------------------------------------ | ---------------- | ---------------------------------------------------- | ---------- | --------- | ---- |
| High-Resolution Image Synthesis with Latent Diffusion Models | LDM              | [arXiv:2112.10752](https://arxiv.org/abs/2112.10752) | 2021.12.20 | CVPR 2022 |      |
| https://stablediffusionweb.com/                              | Stable Diffusion |                                                      |            |           |      |

## Diffusion in image space

| Paper                                                        | Model   | Arxiv                                                | Date       | Comments  | Code |
| ------------------------------------------------------------ | ------- | ---------------------------------------------------- | ---------- | --------- | ---- |
| Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding | Imagen  | [arXiv:2205.11487](https://arxiv.org/abs/2205.11487) | 2022.05.23 | NIPS 2022 |      |
| GLIDE: Towards Photorealistic Image Generation and Editing with Text-Guided Diffusion Models | GLIDE   | [arXiv:2112.10741](https://arxiv.org/abs/2112.10741) | 2021.12.20 | ICML 2022 |      |
| Hierarchical Text-Conditional Image Generation with CLIP Latents | DALL-E2 | [arXiv:2204.06125](https://arxiv.org/abs/2204.06125) | 2022.04.13 |           |      |



# Semantic maps to image

| Paper                                                        | Model      | Arxiv                                                | Date       | Comments  | Code |
| ------------------------------------------------------------ | ---------- | ---------------------------------------------------- | ---------- | --------- | ---- |
| High-Resolution Image Synthesis with Latent Diffusion Models | LDM        | [arXiv:2112.10752](https://arxiv.org/abs/2112.10752) | 2021.12.20 | CVPR 2022 |      |
| Adding Conditional Control to Text-to-Image Diffusion Models | ControlNet | [arXiv:2302.05543](https://arxiv.org/abs/2302.05543) | 2023.09.02 | CVPR 2023 |      |



# Image editing

## Text edit with CLIP

| Paper                                                        | Model           | Arxiv                                                | Date       | Comments  | Code |
| ------------------------------------------------------------ | --------------- | ---------------------------------------------------- | ---------- | --------- | ---- |
| Diffusion Models already have a Semantic Latent Space        | Asyrp           | [arXiv:2210.10960](https://arxiv.org/abs/2210.10960) | 2022.10.20 | ICLR2023  |      |
| DiffusionCLIP: Text-Guided Diffusion Models for Robust Image Manipulation | DiffusionCLIP   | [arXiv:2110.02711](https://arxiv.org/abs/2110.02711) | 2021.10.06 | CVPR2022  |      |
| Blended Diffusion for Text-driven Editing of Natural Images  | Blend Diffusion | [arXiv:2111.14818](https://arxiv.org/abs/2111.14818) | 2021.11.29 | CVPR 2022 |      |
| Hierarchical Text-Conditional Image Generation with CLIP Latents | DALL-E2         | [arXiv:2204.06125](https://arxiv.org/abs/2204.06125) | 2022.04.13 |           |      |



## Exploit big T2I models

| Paper                                                        | Model            | Arxiv                                                | Date       | Comments  | Code |
| :----------------------------------------------------------- | ---------------- | ---------------------------------------------------- | ---------- | --------- | ---- |
| Imagic: Text-Based Real Image Editing with Diffusion Models  | Imagic           | [arXiv:2210.09276](https://arxiv.org/abs/2210.09276) | 2022.10.17 | CVPR 2023 |      |
| FORGEDIT: TEXT GUIDED IMAGE EDITING VIA LEARN- ING AND FORGETTING | Forgedit         | [arXiv:2309.10556](https://arxiv.org/abs/2309.10556) | 2023.09.19 |           |      |
| DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation | DreamBooth       | [arXiv:2208.12242](https://arxiv.org/abs/2208.12242) | 2022.08.25 | CVPR 2023 |      |
| DiffEdit: Diffusion-based semantic image editing with mask guidance | Diffedit         | [arXiv:2210.11427](https://arxiv.org/abs/2210.11427) | 2022.10.20 | ICLR 2023 |      |
| Prompt-to-Prompt Image Editing with Cross Attention Control  | Prompt-to-prompt | [arXiv:2208.01626](https://arxiv.org/abs/2208.01626) | 2022.08.02 | ICLR 2023 |      |

## CFG Inversion Technics

| Paper                                                        | Model               | Arxiv                                                | Date       | Comments  | Code |
| :----------------------------------------------------------- | ------------------- | ---------------------------------------------------- | ---------- | --------- | ---- |
| Direct Inversion: Boosting Diffusion-based Editing with 3 Lines of Code | Direct Inversion    | [arXiv:2310.01506](https://arxiv.org/abs/2310.01506) | 2023.10.02 |           |      |
| StyleDiffusion: Controllable Disentangled Style Transfer via Diffusion Models | Style Diffusion     | [arXiv:2308.07863](https://arxiv.org/abs/2308.07863) | 2023.08.15 | ICCV 2023 |      |
| Null-text Inversion for Editing Real Images using Guided Diffusion Models | Null Text Inversion | [arXiv:2211.09794](https://arxiv.org/abs/2211.09794) | 2022.11.17 | CVPR 2023 |      |
| Negative-prompt Inversion: Fast Image Inversion for Editing with Text-guided Diffusion Models | Negative Inversion  | [arXiv:2305.16807](https://arxiv.org/abs/2305.16807) | 2023.05.26 |           |      |

# Image restoration

# Auxiliary methods for condition injecting