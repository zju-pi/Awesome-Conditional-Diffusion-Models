# A Survey on conditional image synthesis with diffusion model

This repo is constructed for collecting and categorizing papers about conditional image synthesis with-diffusion-model.

## A mind map

![Conditional image synthesis with diffusion model](https://github.com/Szy12345-liv/A-Survey-on-conditional-image-synthesis-with-diffusion-model/blob/main/Images/Conditional%20image%20synthesis%20with%20diffusion%20model.png)

## Contents
- [Refine the generative process](#refine-the-generative-process)
  - [Semantic meaningful Initialization](#better-initialization-for-generative-process)
  - [Projection](#projection)
  - [Guidance](#guidance)
  - [Decomposition-based methods](#decomposition-based-methods)

- [Conditioning the diffusion model](#conditioning-the-diffusion-model)
  - [The architecture conditional diffusion model](#the-architecture-conditional-diffusion-model)
  - [Conditioning on text](#conditioning-on-text)
  - [Conditioning on image](#conditioning-on-image)
  - [Conditioning on visual feature map](#Conditioning-on-visual-feature-map)
  - [Conditioning on intermediate representations](#conditioning-on-intermediate-representations )

- [Exploiting pre-trained text-to-image models](#exploiting-pre-trained-large-conditional-models)
  - [Tabular for conditioning aproaches employed in each work](tabular-for-conditioning-aproaches-employed-in-each-work)
  - [Re-training](#re-training)
    - [Re-training for multi-modal conditional inputs](#re-training-for-multi-modal-conditional-inputs)
    - [Re-training for memorizing the conditional inputs](#re-training-for-memorizing-the-conditional-inputs)

  - [Inversion](#inversion)
  - [Segmentations & Masks](#Segmentations-&-Masks)
  - [Embedding optimization](#embedding-optimization)
  - [Attention manipulation](#Attention manipulation)
  - [Guidance on conditional models](#guidance-on-conditional models)
  - [Other conditioning techniques](#Other conditioning techniques)

- [Augmentations for conditional diffusion model](#augmentations-for-conditional-diffusion-model)
  - [Retrival Augmentation](#Retrival Augmentation)
  - [Composition of conditional diffusion models](composition-of-conditional-diffusion-models)
  - [Sampling acceleration for text-to-image models](#sampling-acceleration-for-text-to-image-models)


**The date in the table represents the publication date of the first version of the paper on Arxiv.**

# Refine the generative process

## Semantic meaningful Initialization

| Model      | Website                                              | Date       | Comments  |
| ---------- | ---------------------------------------------------- | ---------- | --------- |
| **SDEdit** | [arXiv:2108.01073](https://arxiv.org/abs/2108.01073) | 2021.08.02 | ICLR 2022 |
| **DDIB**   | [arXiv:2203.08382](https://arxiv.org/abs/2203.08382) | 2022.03.04 | ICLR 2023 |

## Projection

| Model       | Website                          | Date       | Comments  |
| ----------- | -------------------------------- | ---------- | --------- |
| **ILVR**    | https://arxiv.org/abs/2108.02938 | 2021.08.06 | ICCV 2021 |
| **Repaint** | https://arxiv.org/abs/2201.09865 | 2022.01.24 | CVPR 2022 |
| **CCDF**    | https://arxiv.org/abs/2112.05146 | 2021.12.09 | CVPR 2022 |

## Guidance

| Model                   | Website                                     | Date       | Comments  |
| ----------------------- | ------------------------------------------- | ---------- | --------- |
| **Classifier-Guidance** | https://arxiv.org/abs/2105.05233            | 2021.05.11 | NIPS 2021 |
| **Blend Diffusion**     | https://arxiv.org/abs/2111.14818            | 2021.11.29 | CVPR 2022 |
| **DiffuseIT**           | https://arxiv.org/abs/2209.15264            | 2022.09.30 | ICLR 2023 |
| **SDG**                 | https://arxiv.org/abs/2112.05744            | 2021.12.10 | CVPR 2023 |
| **D2C**                 | https://arxiv.org/abs/2106.06819            | 2021.07.12 | NIPS 2021 |
| **MCG**                 | https://arxiv.org/abs/2206.00941            | 2022.06.02 | NIPS 2022 |
| **DPS**                 | https://arxiv.org/abs/2209.14687            | 2022.09.29 | ILCR 2023 |
| **IIGDM**               | https://openreview.net/forum?id=9_gsMA8MRKQ | 2023.02.02 | ILCR 2023 |
| **GDP**                 | https://arxiv.org/abs/2304.01247            | 2023.04.03 | CVPR 2023 |
| **BlindDPS**            | https://arxiv.org/abs/2211.10656            | 2022.11.29 | CVPR 2023 |
| **FreeDoM**             | https://arxiv.org/abs/2303.09833            | 2023.05.17 | ICCV 2023 |

## Decomposition

| Model     | Website                          | Date       | Comments  |
| --------- | -------------------------------- | ---------- | --------- |
| **SNIPS** | https://arxiv.org/abs/2105.14951 | 2021.05.31 | NIPS 2021 |
| **DDRM**  | https://arxiv.org/abs/2201.11793 | 2022.01.27 | ICLR 2022 |
| **DDNM**  | https://arxiv.org/abs/2212.00490 | 2022.12.01 | ICLR 2023 |

# Conditioning the diffusion model

## Conditioning on text

| Model            | Website                                              | Date       | Comments  |
| ---------------- | ---------------------------------------------------- | ---------- | --------- |
| **VQDM**         | [arXiv:2111.14822](https://arxiv.org/abs/2111.14822) | 2021.11.29 | CVPR 2022 |
| **LDM**          | [arXiv:2112.10752](https://arxiv.org/abs/2112.10752) | 2021.12.20 | CVPR 2022 |
| **GLIDE**        | [arXiv:2112.10741](https://arxiv.org/abs/2112.10741) | 2021.12.20 | ICML 2022 |
| **DALL-E2**      | [arXiv:2204.06125](https://arxiv.org/abs/2204.06125) | 2022.04.13 |           |
| **Imagen**       | [arXiv:2205.11487](https://arxiv.org/abs/2205.11487) | 2022.05.23 | NIPS 2022 |
| **eDiff-I**      | https://arxiv.org/abs/2211.01324                     | 2022.11.02 |           |
| **VQ-Diffusion** | https://arxiv.org/abs/2111.14822                     | 2021.11.29 | CVPR 2022 |

## Conditioning on image

| Model         | Website                          | Date       | Comments        |
| ------------- | -------------------------------- | ---------- | --------------- |
| **SR3**       | https://arxiv.org/abs/2104.07636 | 2021.04.15 | IEEE Trans 2023 |
| **Palette**   | https://arxiv.org/abs/2111.05826 | 2022.11.10 | ICLR 2022       |
| **CDM**       | https://arxiv.org/abs/2106.15282 | 2021.05.30 | ICLR 2022       |
| **SR3+**      | https://arxiv.org/abs/2302.07864 | 2023.01.15 |                 |
| **Unit-DDPM** | https://arxiv.org/abs/2104.05358 | 2021.04.12 |                 |
| **IR-SDE**    | https://arxiv.org/abs/2301.11699 | 2023.01.27 | ICML 2023       |
| **InDI**      | https://arxiv.org/abs/2303.11435 | 2023.03.20 |                 |

## Conditioning on visual feature map

| Model   | Website                          | Date       | Comments |
| ------- | -------------------------------- | ---------- | -------- |
| **SDM** | https://arxiv.org/abs/2207.00050 | 2022.06.30 |          |

## Conditioning on intermediate representations

| Model                     | Website                          | Date       | Comments            |
| ------------------------- | -------------------------------- | ---------- | ------------------- |
| **SRDiff**                | https://arxiv.org/abs/2104.14951 | 2021.04.30 | Neurocomputing 2022 |
| **ResDiff**               | https://arxiv.org/abs/2303.08714 | 2023.03.15 | AAAI 2024           |
| **Diffusion autoencoder** | https://arxiv.org/abs/2111.15640 | 2021.11.30 | CVPR 2022           |

# Exploiting pre-trained text-to-image models

## Tabular for conditioning aproaches employed in each work

| Model    | Website                                              | Date       | Comments | Retraining | Inversion | Segmentations & Masks | Embedding optimization | Attention manipulation | Guidance |
| -------- | ---------------------------------------------------- | ---------- | -------- | ---------- | --------- | --------------------- | ---------------------- | ---------------------- | -------- |
| **PITI** | [arXiv:2205.12952](https://arxiv.org/abs/2205.12952) | 2022.05.25 |          |            |           |                       |                        |                        |          |

## Retraining

### Re-training for multi-modal conditional inputs

| Model                 | Website                                              | Date       | Comments      |
| --------------------- | ---------------------------------------------------- | ---------- | ------------- |
| **PITI**              | [arXiv:2205.12952](https://arxiv.org/abs/2205.12952) | 2022.05.25 |               |
| **Sketch-Guided DM**  | [arXiv:2211.13752](https://arxiv.org/abs/2211.13752) | 2022.11.24 | SIGGRAPH 2023 |
| **T2i-adapter**       | [arXiv:2302.08453](https://arxiv.org/abs/2302.08453) | 2023.02.16 | AAAI 2024     |
| **ControlNet**        | [arXiv:2302.05543](https://arxiv.org/abs/2302.05543) | 2023.09.02 | ICCV 2023     |
| **GLIGEN**            | https://arxiv.org/abs/2301.07093                     | 2023.01.17 | CVPR 2023     |
| **UniControl**        | https://arxiv.org/abs/2305.11147                     | 2023.05.18 | NIPS 2023     |
| **UniControlNet**     | https://arxiv.org/abs/2305.16322                     | 2023.05.25 | NIPS 2023     |
| **InstructPix2Pix**   | https://arxiv.org/abs/2211.09800                     | 2022.11.17 | CVPR 2023     |
| **MoEController**     | https://arxiv.org/abs/2309.04372                     | 2023.09.08 |               |
| **FoI**               | https://arxiv.org/abs/2312.10113                     | 2023.12.15 |               |
| **LOFIE**             | https://arxiv.org/abs/2310.19145                     | 2023.10.29 | EMNLP 2023    |
| **InstructDiffusion** | https://arxiv.org/abs/2309.03895                     | 2023.09.07 |               |
| **Emu Edit**          | https://arxiv.org/abs/2311.10089                     | 2023.11.16 |               |
| **DialogPaint**       | https://arxiv.org/abs/2303.10073                     | 2023.03.17 |               |
| **Inst-Inpaint**      | https://arxiv.org/abs/2304.03246                     | 2023.04.06 |               |
| **HIVE**              | https://arxiv.org/abs/2303.09618                     | 2023.03.16 |               |
| **ImageBrush**        | https://arxiv.org/abs/2308.00906                     | 2023.08.02 | NIPS 2023     |
| **InstructAny2Pix**   | https://arxiv.org/abs/2312.06738                     | 2023.12.11 |               |
| **MGIE**              | https://arxiv.org/abs/2309.17102                     | 2023.09.29 |               |
| **SmartEdit**         | https://arxiv.org/abs/2312.06739                     | 2023.12.11 |               |
| **Imagen Editor**     | https://arxiv.org/abs/2212.06909                     | 2022.12.13 | CVPR 2023     |
| **PbE**               | https://arxiv.org/abs/2211.13227                     | 2022.11.23 | CVPR 2023     |
| **ObjectStitch**      | https://arxiv.org/abs/2212.00932                     | 2022.12.02 | CVPR 2023     |
| **RIC**               | https://arxiv.org/abs/2304.09748                     | 2023.03.31 |               |
| **PhD**               | https://arxiv.org/abs/2306.07596                     | 2023.07.13 |               |
| **Dreaminpainter**    | https://arxiv.org/abs/2312.03771                     | 2023.12.05 |               |
| **Anydoor**           | https://arxiv.org/abs/2307.09481                     | 2023.07.18 |               |
| **Pair-diffusion**    | https://arxiv.org/abs/2303.17546                     | 2023.03.30 | CVPR 2023     |
| **Smartbrush**        | https://arxiv.org/abs/2212.05034                     | 2022.12.09 | CVPR 2023     |
| **IIR-Net**           | https://arxiv.org/abs/2305.17489                     | 2023.05.27 | CVPR 2024     |



### Re-training for personal object memorizing

| Model               | Website                                               | Date       | Comments           |
| ------------------- | ----------------------------------------------------- | ---------- | ------------------ |
| **DreamBooth**      | [arXiv:2208.12242 ](https://arxiv.org/abs/2208.12242) | 2022.08.25 | CVPR 2023          |
| **Imagic**          | [arXiv:2210.09276](https://arxiv.org/abs/2210.09276)  | 2022.10.17 | CVPR 2023          |
| **Unitune**         | https://arxiv.org/abs/2210.09477                      | 2022.10.17 |                    |
| **Forgedit**        | [arXiv:2309.10556](https://arxiv.org/abs/2309.10556)  | 2023.09.19 |                    |
| **LayerDiffusion**  | https://arxiv.org/abs/2305.18676                      | 2023.05.30 | SIGGRAPH ASIA 2023 |
| **CustomDiffusion** | https://arxiv.org/abs/2212.04488                      | 2022.12.08 | CVPR 2023          |
| **E4T**             | https://arxiv.org/abs/2302.12228                      | 2023.02.23 | TOG 2023           |
| **Customedit**      | https://arxiv.org/abs/2305.15779                      | 2023.05.25 |                    |
| **SINE**            | https://arxiv.org/abs/2212.04489                      | 2022.12.08 | CVPR 2023          |

## Inversion

| Model                       | Website                                               | Date       | Comments            |
| --------------------------- | ----------------------------------------------------- | ---------- | ------------------- |
| **Null-Text Inversion**     | [arXiv:2211.09794](https://arxiv.org/abs/2211.09794)  | 2022.11.17 | CVPR 2023           |
| **EDICT**                   | [arXiv:2211.12446](https://arxiv.org/abs/2211.12446)  | 2022.11.22 | CVPR 2023           |
| **Negative Inversion**      | [arXiv:2305.16807 ](https://arxiv.org/abs/2305.16807) | 2023.05.26 |                     |
| **Style Diffusion**         | [arXiv:2308.07863](https://arxiv.org/abs/2308.07863)  | 2023.08.15 | ICCV 2023           |
| **Direct Inversion**        | [arXiv:2310.01506 ](https://arxiv.org/abs/2310.01506) | 2023.10.02 |                     |
| **TF-ICON**                 | https://arxiv.org/abs/2307.12493                      | 2023.06.24 | CVPR 2023           |
| **DDPM Inversion**          | https://arxiv.org/abs/2304.06140                      | 2023.04.12 |                     |
| **Cycle Diffusion**         | https://arxiv.org/abs/2210.05559                      | 2022.10.11 | ICCV 2023           |
| **Prompt Tuning Inversion** | https://arxiv.org/abs/2305.04441                      | 2023.05.08 | CVPR 2023           |
| **KV-Inversion**            | https://arxiv.org/abs/2309.16608                      | 2023.09.28 | PRCV 2023           |
| **StyleDiffusion**          | https://arxiv.org/abs/2308.07863                      | 2023.08.15 | CVPR 2023           |
| **SDE-Drag**                | https://arxiv.org/abs/2311.01410                      | 2023.11.02 | ICLR 2024           |
| **LEDITS++**                | https://arxiv.org/abs/2311.16711                      | 2023.11.28 |                     |
| **FEC**                     | https://arxiv.org/abs/2309.14934                      | 2023.09.26 | ICICML 2023         |
| **EMILIE**                  | https://arxiv.org/abs/2309.00613                      | 2023.09.01 | WCCV 2023           |
| **ProxEdit**                | https://arxiv.org/abs/2306.05414                      | 2023.06.08 | WACV 2024           |
| **Null-Text Guidance**      | https://arxiv.org/abs/2305.06710                      | 2023.05.11 | ACM multimedia 2023 |
| **Fixed-point inversion**   | https://arxiv.org/abs/2312.12540                      | 2023.12.19 |                     |
| **AIDI**                    | https://arxiv.org/abs/2309.04907                      | 2023.09.10 | ICCV 2023           |

## Segmentations & Masks

| Model                      | Website                                               | Date       | Comments  |
| -------------------------- | ----------------------------------------------------- | ---------- | --------- |
| **Diffedit**               | [arXiv:2210.11427 ](https://arxiv.org/abs/2210.11427) | 2022.10.20 | ICLR 2023 |
| **Masactrl**               | https://arxiv.org/abs/2304.08465                      | 2023.04.17 | CVPR 2023 |
| **Object-Shape Variation** | https://arxiv.org/abs/2303.11306                      | 2023.03.20 | CVPR 2023 |
| **DPL**                    | https://arxiv.org/abs/2309.15664                      | 2023.09.27 | NIPS 2024 |
| **Region-Aware Diffusion** | https://arxiv.org/abs/2302.11797                      | 2023.02.23 |           |
| **TDIELR**                 | https://arxiv.org/abs/2311.16432                      | 2023.11.28 |           |

## Embedding optimization

| Model                        | Website                                               | Date       | Comments  |
| ---------------------------- | ----------------------------------------------------- | ---------- | --------- |
| **Textual Inversion**        | [arXiv:2208.01618 ](https://arxiv.org/abs/2208.01618) | 2022.08.02 | ICLR 2023 |
| **ELITE**                    | https://arxiv.org/abs/2302.13848                      | 2023.02.27 | ICCV 2023 |
| **InST**                     | https://arxiv.org/abs/2211.13203                      | 2022.11.23 | CVPR 2023 |
| **PRedItOR**                 | https://arxiv.org/abs/2302.07979                      | 2023.02.15 |           |
| **DiffusionDisentanglement** | https://arxiv.org/abs/2212.08698                      | 2022.12.16 | CVPR 2023 |
| **Imagic**                   | https://arxiv.org/abs/2210.09276                      | 2022.10.17 | CVPR 2023 |
| **Forgedit**                 | https://arxiv.org/abs/2309.10556                      | 2023.09.19 |           |

## Attention manipulation

| Model                | Website                                               | Date       | Comments  |
| -------------------- | ----------------------------------------------------- | ---------- | --------- |
| **Prompt-to-prompt** | [arXiv:2208.01626 ](https://arxiv.org/abs/2208.01626) | 2022.08.02 | ICLR 2023 |
| **Pix2Pix-Zero**     | https://arxiv.org/abs/2302.03027                      | 2023.02.06 |           |
| **MasaCtrl**         | https://arxiv.org/abs/2304.08465                      | 2023.04.17 | CVPR 2023 |
| **Plug-and-play**    | https://arxiv.org/abs/2211.12572                      | 2022.11.22 | CVPR 2023 |
| **TF-ICON**          | https://arxiv.org/abs/2307.12493                      | 2023.07.24 | CVPR 2023 |
| **EBMs**             | https://arxiv.org/abs/2306.09869                      | 2023.06.16 | NIPS 2024 |
| **P2Plus**           | https://arxiv.org/abs/2308.07863                      | 2023.08.15 | CVPR 2023 |
| **FADING**           | https://arxiv.org/abs/2309.11321                      | 2023.09.20 | BMCV 2023 |
| **DPL**              | https://arxiv.org/abs/2309.15664                      | 2023.09.27 | NIPS 2024 |
| **CustomEdit**       | https://arxiv.org/abs/2305.15779                      | 2023.05.25 |           |
| **Rediffuser**       | https://arxiv.org/abs/2305.04651                      | 2023.05.08 |           |

## Guidance

| Model                         | Website                          | Date       | Comments           |
| ----------------------------- | -------------------------------- | ---------- | ------------------ |
| **Sketch Guided diffusion**   | https://arxiv.org/abs/2211.13752 | 2022.11.24 |                    |
| **GradOP**                    | https://arxiv.org/abs/2311.10709 | 2023.11.17 |                    |
| **Self-Guidance**             | https://arxiv.org/abs/2306.00986 | 2023.06.01 | NIPS 2023          |
| **Layout Control Guidance**   | https://arxiv.org/abs/2304.03373 | 2023.04.06 |                    |
| **Dragondiffusion**           | https://arxiv.org/abs/2307.02421 | 2023.07.05 | ICLR 2024          |
| **Universal  guidance**       | https://arxiv.org/abs/2302.07121 | 2023.02.14 | CVPR 2023          |
| **FreeDoM **                  | https://arxiv.org/abs/2303.09833 | 2023.03.17 | ICCV 2023          |
| **Classifier-free  guidance** | https://arxiv.org/abs/2207.12598 | 2022.0726  | NIPS 2022 workshop |

## Other Conditioning Techniques

| Model   | Website                          | Date       | Comments  |
| ------- | -------------------------------- | ---------- | --------- |
| **DDS** | https://arxiv.org/abs/2304.07090 | 2023.04.14 | ICCV 2023 |
| **CDS** | https://arxiv.org/abs/2311.18608 | 2023.11.30 |           |

# Augmentations for conditional diffusion model

## Retrival Augmentation

| Model             | Website                                     | Date       | Comments  |
| ----------------- | ------------------------------------------- | ---------- | --------- |
| **RDM**           | https://openreview.net/forum?id=Bqk9c0wBNrZ | 2022.11.01 | NIPS 2022 |
| **KNN-Diffusion** | https://arxiv.org/abs/2204.02849            | 2022.04.06 | ICLR 2023 |
| **Re-Imagen**     | https://arxiv.org/abs/2209.14491            | 2022.09.29 | ICLR 2023 |

## Composition of Conditional Diffusion Models

| Model                           | Website                          | Date       | Comments  |
| ------------------------------- | -------------------------------- | ---------- | --------- |
| **Composable Diffusion Models** | https://arxiv.org/abs/2206.01714 | 2022.06.03 | ECCV 2022 |
| **Multi-Diffusion**             | https://arxiv.org/abs/2302.08113 | 2023.02.16 | ICML 2023 |

## Sampling acceleration for conditional diffusion models

| Model                   | Website                          | Date       | Comments  |
| ----------------------- | -------------------------------- | ---------- | --------- |
| **CoDi**                | https://arxiv.org/abs/2310.01407 | 2023.10.02 | CVPR 2024 |
| **Distillation of GDM** | https://arxiv.org/abs/2210.03142 | 2022.10.06 | CVPR 2023 |
