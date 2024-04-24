## VideoCraftXtend: AI-Enhanced Text-to-Video Generation with Extended Length and Enhanced Motion Smoothness

Check our online demo here: <a href='https://huggingface.co/spaces/ychenhq/VideoCrafterXtend'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue'></a>
GitHub page: <a href='https://github.com/chloeleehn/VideoCraftXtend'>
 
## Introduction
VideoCraftXtend is an open-source video generation and editing toolbox for crafting video content.
This project aims to tackle challenges in T2V generation, specifically focusing on the production of long videos, enhancing motion smoothness quality and improving content diversity. We propose a comprehensive framework that integrates a T2V diffusion model, utilizes the OpenAI GPT API, incorporates a Video Quality Assessment (VQA) model, and refines an Interpolation model. 

### 1. Generic Text-to-video Generation
Click on the text to access the video.

<table class="center">
 <tr> 
  <td style="text-align:center;" width="320">"Original prompt"</td>
  <td style="text-align:center;" width="320">"AI-modified prompt"</td>
  <td style="text-align:center;" width="320">"AI-modified prompt"</td>
  <td style="text-align:center;" width="320">"Choosen video after Interpolation"</td>
 </tr>
  <tr>
   <td style="text-align:center;" width="320"><a href="https://github.com/chloeleehn/VideoCraftXtend/blob/main/VideoCrafter/results/cat/0001.mp4">"There is a cat dancing on the sand."</a></td>
   <td style="text-align:center;" width="320"><a href="https://github.com/chloeleehn/VideoCraftXtend/blob/main/VideoCrafter/results/cat/0002.mp4">"Behold the mesmerizing sight of a cat elegantly dancing amidst the soft grains of sand."</a></td>
   <td style="text-align:center;" width="320"><a href="https://github.com/chloeleehn/VideoCraftXtend/blob/main/VideoCrafter/results/cat/0003.mp4">"The fluffy cat is joyfully prancing and twirling on the soft golden sand, its elegant movements mirroring the peaceful seaside setting."</a></td>
   <td style="text-align:center;" width="320"><a href="https://github.com/chloeleehn/VideoCraftXtend/blob/main/VideoCrafter/results/cat/0002_4X_16fps.mp4">"Behold the mesmerizing sight of a cat elegantly dancing amidst the soft grains of sand."</a></td>
  </tr>
</table >


## Setup

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


## Inference 
### 1. Text-to-Video local Gradio demo
1) Open `VideoCraftXtend.ipynb`, run the cells till generating Gradio Interface.
2) Input prompt, customize the parameters and get the resulting video
3) The last section of the file is evaluation results been put in our report)
4) Open the `VideoCraftXtend.ipynb` notebook and run the cells until you reach the point where the Gradio interface is generated.
5) Once the Gradio interface is generated, you can input prompts and customize the parameters according to your requirements. The resulting video should be generated within an estimated timeframe of 15-20 minutes.
6) The last section of `VideoCraftXtend.ipynb` contains the evaluation results that were included in our report.


## 📋 Techinical Report
[VideoCraftXtend: AI-Enhanced Text-to-Video Generation with Extended Length and Enhanced Motion Smoothness](https://github.com/chloeleehn/VideoCraftXtend/blob/main/VideoCraftXtend_report.pdf)



## Acknowledgements
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
