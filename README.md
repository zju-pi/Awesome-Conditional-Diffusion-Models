# A Survey on conditional image synthesis with diffusion model

This repo is constructed for collecting and categorizing papers about conditional image synthesis with-diffusion-model.

## A mind map

![Conditional image synthesis with diffusion model](https://github.com/Szy12345-liv/A-Survey-on-conditional-image-synthesis-with-diffusion-model/blob/main/Images/Conditional%20image%20synthesis%20with%20diffusion%20model.png)

## Contents
- [Domain Translation](#domain-translation)
- [Text-to-image](#text-to-image)
  - [Diffusion in latent space](#diffusion-in-latent-space)
  - [Diffusion in image space](#diffusion-in-image-space)
- [Semantic maps to image](#semantic-maps-to-image)
- [Image editing](#image-editing)
  - [Text edit with CLIP](#text-edit-with-CLIP)
  - [Exploit big T2I models](#exploit-big-T2I-models)
  - [CFG Inversion Technics](#CFG-inversion-technics)
  
- [Customization](#customization)
  - [Model fine tuning](#model-fine-tuning)
  
  - [Textual embedding optimization](#textual-embedding-optimization)
  
- [Image restoration](#image-restoration)
  - [Supervise](#supervise)
  - [Zero-shot](#zero-shot)
- [Auxiliary methods for condition injecting](#auxiliary-methods-for-condition-injecting)

**The date in the table represents the publication date of the first version of the paper on Arxiv.**

# Domain Translation

| Paper                                                        | Model      | Arxiv                                                | Date       | Comments  | Code |
| ------------------------------------------------------------ | ---------- | ---------------------------------------------------- | ---------- | --------- | ---- |
| SDEdit: Guided Image Synthesis and Editing with  Stochastic Differential Equations | **SDEdit** | [arXiv:2108.01073](https://arxiv.org/abs/2108.01073) | 2021.08.02 | ICLR 2022 |      |
| ILVR: Conditioning Method for Denoising Diffusion  Probabilistic Models | **ILVR**   | [arXiv:2108.02938](https://arxiv.org/abs/2108.02938) | 2021.08.06 | ICCV 2021 |      |
| Dual Diffusion Implicit Bridges for Image-to-Image  Translation | **DDIB**   | [arXiv:2203.08382](https://arxiv.org/abs/2203.08382) | 2022.03.04 | ICLR 2023 |      |



# Text-to-image

## Diffusion in latent Space

| Paper                                                        | Model            | Arxiv                                                | Date       | Comments  | Code |
| ------------------------------------------------------------ | ---------------- | ---------------------------------------------------- | ---------- | --------- | ---- |
| High-Resolution Image Synthesis with Latent Diffusion Models | **LDM**          | [arXiv:2112.10752](https://arxiv.org/abs/2112.10752) | 2021.12.20 | CVPR 2022 |      |
| https://stablediffusionweb.com/                              | Stable Diffusion |                                                      |            |           |      |

## Diffusion in image space

| Paper                                                        | Model       | Arxiv                                                | Date       | Comments  | Code |
| ------------------------------------------------------------ | ----------- | ---------------------------------------------------- | ---------- | --------- | ---- |
| GLIDE: Towards Photorealistic Image Generation and  Editing with Text-Guided Diffusion Models | **GLIDE**   | [arXiv:2112.10741](https://arxiv.org/abs/2112.10741) | 2021.12.20 | ICML 2022 |      |
| Hierarchical Text-Conditional Image Generation with  CLIP Latents | **DALL-E2** | [arXiv:2204.06125](https://arxiv.org/abs/2204.06125) | 2022.04.13 |           |      |
| Photorealistic Text-to-Image Diffusion Models with  Deep Language Understanding | **Imagen**  | [arXiv:2205.11487](https://arxiv.org/abs/2205.11487) | 2022.05.23 | NIPS 2022 |      |



# Semantic maps to image

| Paper                                                        | Model          | Arxiv                                                | Date       | Comments  | Code |
| ------------------------------------------------------------ | -------------- | ---------------------------------------------------- | ---------- | --------- | ---- |
| High-Resolution Image Synthesis with Latent Diffusion Models | **LDM**        | [arXiv:2112.10752](https://arxiv.org/abs/2112.10752) | 2021.12.20 | CVPR 2022 |      |
| Adding Conditional Control to Text-to-Image Diffusion Models | **ControlNet** | [arXiv:2302.05543](https://arxiv.org/abs/2302.05543) | 2023.09.02 | CVPR 2023 |      |



# Image editing

## Text edit with CLIP

| Paper                                                        | Model               | Arxiv                                                | Date       | Comments  | Code |
| ------------------------------------------------------------ | ------------------- | ---------------------------------------------------- | ---------- | --------- | ---- |
| DiffusionCLIP: Text-Guided Diffusion Models for Robust  Image Manipulation | **DiffusionCLIP**   | [arXiv:2110.02711](https://arxiv.org/abs/2110.02711) | 2021.10.06 | CVPR2022  |      |
| Blended Diffusion for Text-driven Editing of Natural  Images | **Blend Diffusion** | [arXiv:2111.14818](https://arxiv.org/abs/2111.14818) | 2021.11.29 | CVPR 2022 |      |
| Hierarchical Text-Conditional Image Generation with  CLIP Latents | **DALL-E2**         | [arXiv:2204.06125](https://arxiv.org/abs/2204.06125) | 2022.04.13 |           |      |
| Diffusion Models already have a Semantic Latent Space        | **Asyrp**           | [arXiv:2210.10960](https://arxiv.org/abs/2210.10960) | 2022.10.20 | ICLR2023  |      |



## Exploit big T2I models

| Paper                                                        | Model                | Arxiv                                                 | Date       | Comments  | Code |
| :----------------------------------------------------------- | -------------------- | ----------------------------------------------------- | ---------- | --------- | ---- |
| Prompt-to-Prompt Image Editing with Cross Attention  Control | **Prompt-to-prompt** | [arXiv:2208.01626 ](https://arxiv.org/abs/2208.01626) | 2022.08.02 | ICLR 2023 |      |
| Imagic: Text-Based Real Image Editing with Diffusion  Models | **Imagic**           | [arXiv:2210.09276](https://arxiv.org/abs/2210.09276)  | 2022.10.17 | CVPR 2023 |      |
| DiffEdit: Diffusion-based semantic image editing with  mask guidance | **Diffedit**         | [arXiv:2210.11427 ](https://arxiv.org/abs/2210.11427) | 2022.10.20 | ICLR 2023 |      |
| FORGEDIT: TEXT GUIDED IMAGE EDITING VIA LEARN- ING AND  FORGETTING | **Forgedit**         | [arXiv:2309.10556](https://arxiv.org/abs/2309.10556)  | 2023.09.19 |           |      |

## CFG Inversion Technics

| Paper                                                        | Model                   | Arxiv                                                 | Date       | Comments  | Code |
| :----------------------------------------------------------- | ----------------------- | ----------------------------------------------------- | ---------- | --------- | ---- |
| Null-text Inversion for Editing Real Images using  Guided Diffusion Models | **Null Text Inversion** | [arXiv:2211.09794](https://arxiv.org/abs/2211.09794)  | 2022.11.17 | CVPR 2023 |      |
| EDICT: Exact Diffusion Inversion via Coupled Transformations | **EDICT**               | [arXiv:2211.12446](https://arxiv.org/abs/2211.12446)  | 2022.11.22 | CVPR 2023 |      |
| Negative-prompt Inversion: Fast Image Inversion for  Editing with Text-guided Diffusion Models | **Negative Inversion**  | [arXiv:2305.16807 ](https://arxiv.org/abs/2305.16807) | 2023.05.26 |           |      |
| StyleDiffusion: Controllable Disentangled Style  Transfer via Diffusion Models | Style Diffusion         | [arXiv:2308.07863](https://arxiv.org/abs/2308.07863)  | 2023.08.15 | ICCV 2023 |      |
| Direct Inversion: Boosting Diffusion-based Editing  with 3 Lines of Code | Direct Inversion        | [arXiv:2310.01506 ](https://arxiv.org/abs/2310.01506) | 2023.10.02 |           |      |

# Customization

## Model fine tuning

| Paper                                                        | Model          | Arxiv                                                 | Date       | Comments  | Code |
| :----------------------------------------------------------- | -------------- | ----------------------------------------------------- | ---------- | --------- | ---- |
| DreamBooth: Fine Tuning Text-to-Image Diffusion Models  for Subject-Driven Generation | **DreamBooth** | [arXiv:2208.12242 ](https://arxiv.org/abs/2208.12242) | 2022.08.25 | CVPR 2023 |      |

## Textual embedding optimization

| Paper                                                        | Model                 | Arxiv                                                 | Date       | Comments  | Code |
| :----------------------------------------------------------- | --------------------- | ----------------------------------------------------- | ---------- | --------- | ---- |
| An Image is Worth One Word: Personalizing Text-to-Image Generation using Textual Inversion | **Textual Inversion** | [arXiv:2208.01618 ](https://arxiv.org/abs/2208.01618) | 2022.08.02 | ICLR 2023 |      |

# Image restoration

## Supervise

| Paper                                                        | Model                       | Arxiv                                                 | Date       | Comments            | Code |
| :----------------------------------------------------------- | --------------------------- | ----------------------------------------------------- | ---------- | ------------------- | ---- |
| Image Super-Resolution via Iterative Refinement              | **SR3**                     | [arXiv:2104.07636](https://arxiv.org/abs/2104.07636)  | 2021.04.15 |                     |      |
| SRDiff: Single Image Super-Resolution with Diffusion  Probabilistic Models | **SRDiff**                  | [arXiv:2104.14951 ](https://arxiv.org/abs/2104.14951) | 2021.04.30 | Neurocomputing 2022 |      |
| Cascaded Diffusion Models for High Fidelity Image  Generation | **Cascade Diffusion Model** | [arXiv:2106.15282 ](https://arxiv.org/abs/2106.15282) | 2021.05.30 | JMLR 2022           |      |
| Palette: Image-to-Image Diffusion Models                     | **Palette**                 | [arXiv:2111.05826 ](https://arxiv.org/abs/2111.05826) | 2021.11.10 | ICLR 2022           |      |
| Image Restoration with Mean-Reverting Stochastic  Differential Equations | **IR-SDE**                  | [arXiv:2301.11699 ](https://arxiv.org/abs/2301.11699) | 2023.01.27 | ICML 2023           |      |
| Denoising Diffusion Probabilistic Models for Robust  Image Super-Resolution in the Wild | **SR3+**                    | [arXiv:2302.07864 ](https://arxiv.org/abs/2302.07864) | 2023.02.15 |                     |      |
| ResDiff: Combining CNN and Diffusion Model for Image  Super-Resolution | **Resdiff**                 | [arXiv:2303.08714 ](https://arxiv.org/abs/2303.08714) | 2023.03.15 |                     |      |

## Zero-shot

| Paper                                                        | Model               | Arxiv                                                 | Date       | Comments  | Code |
| :----------------------------------------------------------- | ------------------- | ----------------------------------------------------- | ---------- | --------- | ---- |
| Blended Diffusion for Text-driven Editing of Natural  Images | **Blend Diffusion** | [arXiv:2111.14818 ](https://arxiv.org/abs/2111.14818) | 2021.11.29 | CVPR 2022 |      |
| RePaint: Inpainting using Denoising Diffusion  Probabilistic Models | **Repaint**         | [arXiv:2201.09865 ](https://arxiv.org/abs/2201.09865) | 2022.01.24 | CVPR 2022 |      |
| Denoising Diffusion Restoration Models                       | **DDRM**            | [arXiv:2201.11793 ](https://arxiv.org/abs/2201.11793) | 2022.01.27 | NIPS 2022 |      |
| Diffusion Posterior Sampling for General Noisy Inverse  Problems | **DPS**             | [arXiv:2209.14687 ](https://arxiv.org/abs/2209.14687) | 2022.09.29 | ICLR 2023 |      |
| Parallel Diffusion Models of Operator and Image for  Blind Inverse Problems | BlindDPS            | [arXiv:2211.10656 ](https://arxiv.org/abs/2211.10656) | 2022.11.19 | CVPR 2023 |      |

# Auxiliary methods for condition injecting

| Paper                                                        | Model                 | Arxiv                                                 | Date       | Comments            | Code |
| :----------------------------------------------------------- | --------------------- | ----------------------------------------------------- | ---------- | ------------------- | ---- |
| BLIP: Bootstrapping Language-Image Pre-training for  Unified Vision-Language Understanding and Generation | BLIP                  | [arXiv:2201.12086 ](https://arxiv.org/abs/2201.12086) | 2022.01.28 | ICML 2022           |      |
| Classifier-Free Diffusion Guidance                           | **CFG**               | [arXiv:2207.12598](https://arxiv.org/abs/2207.12598)  | 2022.07.26 | NIPS 2021  Workshop |      |
| An Image is Worth One Word: Personalizing  Text-to-Image Generation using Textual Inversion | **Textual Inversion** | [arXiv:2208.01618 ](https://arxiv.org/abs/2208.01618) | 2022.08.02 | ICLR 2023           |      |
