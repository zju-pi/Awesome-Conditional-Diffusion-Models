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
| **InstructPix2Pix**   | https://arxiv.org/abs/2211.09800                     | 2022.11.17 |               |
| **MoEController**     |                                                      |            |               |
| **FoI**               |                                                      |            |               |
| **LOFIE**             |                                                      |            |               |
| **InstructDiffusion** |                                                      |            |               |
| **Emu Edit**          |                                                      |            |               |
| **DialogPaint**       |                                                      |            |               |
| **Inst-Inpaint**      |                                                      |            |               |
| **HIVE**              |                                                      |            |               |
| **ImageBrush**        |                                                      |            |               |
| **InstructAny2Pix**   |                                                      |            |               |
| **MGIE**              |                                                      |            |               |
| **SmartEdit**         |                                                      |            |               |
| **Imagen Editor**     |                                                      |            |               |
| **PbE**               |                                                      |            |               |
| **ObjectStitch**      |                                                      |            |               |
| **RIC**               |                                                      |            |               |
| **PhD**               |                                                      |            |               |
| **Dreaminpainter**    |                                                      |            |               |
| **Anydoor**           |                                                      |            |               |
| **Pair-diffusion**    |                                                      |            |               |
| **Smartbrush**        |                                                      |            |               |
| **IIR-Net**           |                                                      |            |               |



### Re-training for personal object memorizing

| Model               | Website                                               | Date       | Comments  |
| ------------------- | ----------------------------------------------------- | ---------- | --------- |
| **DreamBooth**      | [arXiv:2208.12242 ](https://arxiv.org/abs/2208.12242) | 2022.08.25 | CVPR 2023 |
| **Imagic**          | [arXiv:2210.09276](https://arxiv.org/abs/2210.09276)  | 2022.10.17 | CVPR 2023 |
| **Unitune**         |                                                       |            |           |
| **Forgedit**        | [arXiv:2309.10556](https://arxiv.org/abs/2309.10556)  | 2023.09.19 |           |
| **LayerDiffusion**  |                                                       |            |           |
| **CustomDiffusion** |                                                       |            |           |
| **E4T**             |                                                       |            |           |
| **Customedit**      |                                                       |            |           |
| **SINE**            |                                                       |            |           |

## Inversion

| Model                       | Website                                               | Date       | Comments  |
| --------------------------- | ----------------------------------------------------- | ---------- | --------- |
| **Null-Text Inversion**     | [arXiv:2211.09794](https://arxiv.org/abs/2211.09794)  | 2022.11.17 | CVPR 2023 |
| **EDICT**                   | [arXiv:2211.12446](https://arxiv.org/abs/2211.12446)  | 2022.11.22 | CVPR 2023 |
| **Negative Inversion**      | [arXiv:2305.16807 ](https://arxiv.org/abs/2305.16807) | 2023.05.26 |           |
| **Style Diffusion**         | [arXiv:2308.07863](https://arxiv.org/abs/2308.07863)  | 2023.08.15 | ICCV 2023 |
| **Direct Inversion**        | [arXiv:2310.01506 ](https://arxiv.org/abs/2310.01506) | 2023.10.02 |           |
| **TF-ICON**                 |                                                       |            |           |
| **DDPM Inversion**          |                                                       |            |           |
| **Cycle Diffusion**         |                                                       |            |           |
| **Prompt Tuning Inversion** |                                                       |            |           |
| **KV-Inversion**            |                                                       |            |           |
| **StyleDiffusion**          |                                                       |            |           |
| **SDE-Drag**                |                                                       |            |           |
| **LEDITS++**                |                                                       |            |           |
| **FEC**                     |                                                       |            |           |
| **EMILIE**                  |                                                       |            |           |
| **ProxEdit**                |                                                       |            |           |
| **Null-Text Guidance**      |                                                       |            |           |
| **Fixed-point inversion**   |                                                       |            |           |
| **AIDI**                    |                                                       |            |           |

## Segmentations & Masks

| Model                      | Website                                               | Date       | Comments  |
| -------------------------- | ----------------------------------------------------- | ---------- | --------- |
| **Diffedit**               | [arXiv:2210.11427 ](https://arxiv.org/abs/2210.11427) | 2022.10.20 | ICLR 2023 |
| **Masactrl**               |                                                       |            |           |
| **Object-Shape Variation** |                                                       |            |           |
| **DPL**                    |                                                       |            |           |
| **Region-Aware Diffusion** |                                                       |            |           |
| **TDILER**                 |                                                       |            |           |

## Embedding optimization

| Model                        | Website                                               | Date       | Comments  |
| ---------------------------- | ----------------------------------------------------- | ---------- | --------- |
| **Textual Inversion**        | [arXiv:2208.01618 ](https://arxiv.org/abs/2208.01618) | 2022.08.02 | ICLR 2023 |
| **ELITE**                    |                                                       |            |           |
| **InST**                     |                                                       |            |           |
| **PRedItOR**                 |                                                       |            |           |
| **DiffusionDisentanglement** |                                                       |            |           |
| **Imagic**                   |                                                       |            |           |
| **Forgedit**                 |                                                       |            |           |

## Attention manipulation

| Model                | Website                                               | Date       | Comments  |
| -------------------- | ----------------------------------------------------- | ---------- | --------- |
| **Prompt-to-prompt** | [arXiv:2208.01626 ](https://arxiv.org/abs/2208.01626) | 2022.08.02 | ICLR 2023 |
| **Pix2Pix-Zero**     |                                                       |            |           |
| **MasaCtrl**         |                                                       |            |           |
| **Plug-and-play**    |                                                       |            |           |
| **TF-ICON**          |                                                       |            |           |
| **EBMs**             |                                                       |            |           |
| **P2Plus**           |                                                       |            |           |
| **FADING**           |                                                       |            |           |
| **EBMs**             |                                                       |            |           |
| **DPL**              |                                                       |            |           |
| **CustomEdit**       |                                                       |            |           |
| **Rediffuser**       |                                                       |            |           |

## Guidance

| Model                         | Website                          | Date      | Comments           |
| ----------------------------- | -------------------------------- | --------- | ------------------ |
| **Sketch Guided diffusion**   |                                  |           |                    |
| **GradOP**                    |                                  |           |                    |
| **Self-Guidance**             |                                  |           |                    |
| **Layout Control Guidance**   |                                  |           |                    |
| **Dragondiffusion**           |                                  |           |                    |
| **Universal  guidance**       |                                  |           |                    |
| **FreeDoM **                  |                                  |           |                    |
| **Classifier-free  guidance** | https://arxiv.org/abs/2207.12598 | 2022.0726 | NIPS 2022 workshop |

## Other Conditioning Techniques

| Model   | Website | Date | Comments |
| ------- | ------- | ---- | -------- |
| **DDS** |         |      |          |
| **CDS** |         |      |          |

# Augmentations for conditional diffusion model

## Retrival Augmentation

| Model             | Website | Date | Comments |
| ----------------- | ------- | ---- | -------- |
| **RDM**           |         |      |          |
| **KNN-Diffusion** |         |      |          |
| **Re-Imagen**     |         |      |          |

## Composition of Conditional Diffusion Models

| Model                           | Website | Date | Comments |
| ------------------------------- | ------- | ---- | -------- |
| **Composable Diffusion Models** |         |      |          |
| **Multi-Diffusion**             |         |      |          |

## Sampling acceleration for conditional diffusion models

| Model                   | Website | Date |
| ----------------------- | ------- | ---- |
| **CoDi**                |         |      |
| **Distillation of GDM** |         |      |
