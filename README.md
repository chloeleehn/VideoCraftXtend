## *** VideoCraftXtend: AI-Enhanced Text-to-Video Generation with Extended Length and Enhanced Motion Smoothness ***

<a href='https://huggingface.co/spaces/ychenhq/VideoCrafterXen'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue'></a>

------
 
## Introduction
VideoCraftXtend is an open-source video generation and editing toolbox for crafting video content.
This project aims to tackle challenges in T2V generation, specifically focusing on the production of long videos, enhancing motion smoothness quality and improving content diversity. We propose a comprehensive framework that integrates a T2V diffusion model, utilizes the OpenAI GPT API, incorporates a Video Quality Assessment (VQA) model, and refines an Interpolation model. 

### 1. Generic Text-to-video Generation
Click the GIF to access the high-resolution video.

<table class="center">
  <td><a href="https://github.com/AILab-CVC/VideoCrafter/assets/18735168/d20ee09d-fc32-44a8-9e9a-f12f44b30411"><img src=assets/t2v/tom.gif width="320"></td>
  <td><a href="https://github.com/AILab-CVC/VideoCrafter/assets/18735168/f1d9f434-28e8-44f6-a9b8-cffd67e4574d"><img src=assets/t2v/child.gif width="320"></td>
  <td><a href="https://github.com/AILab-CVC/VideoCrafter/assets/18735168/bbcfef0e-d8fb-4850-adc0-d8f937c2fa36"><img src=assets/t2v/woman.gif width="320"></td>
  <tr>
  <td style="text-align:center;" width="320">"Tom Cruise's face reflects focus, his eyes filled with purpose and drive."</td>
  <td style="text-align:center;" width="320">"A child excitedly swings on a rusty swing set, laughter filling the air."</td>
  <td style="text-align:center;" width="320">"A young woman with glasses is jogging in the park wearing a pink headband."</td>
  <tr>
</table >

<table class="center">
  <td><a href="https://github.com/AILab-CVC/VideoCrafter/assets/18735168/7edafc5a-750e-45f3-a46e-b593751a4b12"><img src=assets/t2v/couple.gif width="320"></td>
  <td><a href="https://github.com/AILab-CVC/VideoCrafter/assets/18735168/37fe41c8-31fb-4e77-bcf9-fa159baa6d86"><img src=assets/t2v/rabbit.gif width="320"></td>
  <td><a href="https://github.com/AILab-CVC/VideoCrafter/assets/18735168/09791a46-a243-41b8-a6bb-892cdd3a83a2"><img src=assets/t2v/duck.gif width="320"></td>
  <tr>
  <td style="text-align:center;" width="320">"With the style of van gogh, A young couple dances under the moonlight by the lake."</td>
  <td style="text-align:center;" width="320">"A rabbit, low-poly game art style"</td>
  <td style="text-align:center;" width="320">"Impressionist style, a yellow rubber duck floating on the wave on the sunset"</td>
  <tr>
</table >


## ⚙️ Setup

### 1. Install Environment
1) Via Anaconda
   ```bash
   conda create -n videocraftxtend python=3.8.5
   conda activate videocraftxtend
   pip install -r requirements.txt
   ```
2) Using Google Colab Pro

### 2. Download the model checkpoints
1) Download pretrained T2V models via [Hugging Face](https://huggingface.co/VideoCrafter/VideoCrafter2/blob/main/model.ckpt), and put the `model.ckpt` in `VideoCrafter/checkpoints/base_512_v2/model.ckpt`.
2) Download pretrained Interpolation models viea [Google Drive](https://drive.google.com/drive/folders/1TBEwF2PmSGyDngP1anjNswlIfwGh2NzU?usp=sharing), and put the `flownet.pkl` in `VideoCrafter/ECCV2022-RIFE/train_log/flownet.pkl`.

## 💫 Inference 
### 1. Text-to-Video local Gradio demo
1) Open `VideoCraftXtend.ipynb`, run the cells till generating Gradio Interface.
2) Input prompt, customize the parameters and get the resulting video
3) The last section of the file is evaluation results been put in our report)
4) Open the `VideoCraftXtend.ipynb` notebook and run the cells until you reach the point where the Gradio interface is generated.
5) Once the Gradio interface is generated, you can input prompts and customize the parameters according to your requirements. The resulting video should be generated within an estimated timeframe of 15-20 minutes.
6) The last section of `VideoCraftXtend.ipynb` contains the evaluation results that were included in our report.


---
## 📋 Techinical Report
😉 VideoCrafter2 Tech report: [VideoCrafter2: Overcoming Data Limitations for High-Quality Video Diffusion Models](https://arxiv.org/abs/2401.09047)


## 🤗 Acknowledgements
Our codebase builds on 
1) [Stable Diffusion](https://github.com/Stability-AI/stablediffusion)
2) [VideoCrafter2](https://github.com/AILab-CVC/VideoCrafter)
3) [UVQ](https://github.com/google/uvq)
4) [VBench](https://github.com/Vchitect/VBench)
5) [RIFE](https://github.com/hzwer/ECCV2022-RIFE)
Thanks the authors for sharing their codebases! 


## 📢 Disclaimer
We develop this repository for RESEARCH purposes, so it can only be used for personal/research/non-commercial purposes.
****
