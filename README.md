# FLUX-Makeup: High-Fidelity, Identity-Consistent, and Robust Makeup Transfer via Diffusion Transformer

[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-FLUX_Makeup-ff9900?style=flat)](https://huggingface.co/qihoo360/FLUX-Makeup) [![arXiv](https://img.shields.io/badge/arXiv-2511.00956-B31B1B?style=flat)](https://arxiv.org/abs/2508.05069)

![examples](assets/example.png)

We propose **FLUX-Makeup**, a high-fidelity, identityconsistent, and robust makeup transfer framework that eliminates the need for any auxiliary face-control components.

## 💪 Highlight Feature


- **Core Strengths:** FLUX-Makeup delivers high-fidelity, consistent, and robust makeup transfer using only a 'source + reference' input—the most natural interaction format—without the need for any additional facial control modules.
- **Data Engine & HQMT Dataset:** We developed an extensible, filterable, and quality-controlled data generation pipeline, and curated a high-quality paired makeup dataset containing over 50,000 samples, named HQMT.
- **Architecture: Decoupled Feature Injection:** Through RefLoRAInjector, we define two sets of low-rank projections to achieve precise extraction of makeup-related information while effectively preventing identity collapse and background distortion.

## 🧩 Environment Setup


```
pip install -r requirements.txt
```

## ⏳ Inference Pipeline

Here we provide the inference code for our FLUX-Makeup.

  

  

```
sh eval.sh
```


## 🌸 Acknowledgement

  

  

This code is mainly built upon [Diffusers](https://github.com/huggingface/diffusers/tree/main), [Flux](https://github.com/huggingface/diffusers/tree/main/src/diffusers/pipelines/flux), [Stable-Makeup](https://github.com/Xiaojiu-z/Stable-Makeup) and [ComfyUI_Stable_Makeup](https://github.com/smthemex/ComfyUI_Stable_Makeup) repositories. Thanks so much for their solid work!




If you find this repository useful, please consider citing our paper:
```
@article{zhu2025flux,
  title={FLUX-Makeup: High-Fidelity, Identity-Consistent, and Robust Makeup Transfer via Diffusion Transformer},
  author={Zhu, Jian and Liu, Shanyuan and Li, Liuzhuozheng and Gong, Yue and Wang, He and Cheng, Bo and Ma, Yuhang and Wu, Liebucha and Wu, Xiaoyu and Leng, Dawei and others},
  journal={arXiv preprint arXiv:2508.05069},
  year={2025}
}

```

  
