### (CVPR 2026) STCDiT: Spatio-Temporally Consistent Diffusion Transformer for High-Quality Video Super-Resolution
> ![visitors](https://visitor-badge.laobi.icu/badge?page_id=JyChen9811/STCDiT)

> [[Project Page](https://jychen9811.github.io/STCDiT_page/)] &emsp; [[Paper](https://arxiv.org/pdf/2511.18786)]


> [Junyang Chen](https://jychen9811.github.io/), [Jiangxin Dong](https://scholar.google.com/citations?user=ruebFVEAAAAJ&hl=zh-CN&oi=ao), [Long Sun](https://sunny2109.github.io/), [Yixin Yang](https://yyang181.github.io/), [Jinshan Pan](https://jspan.github.io/),  <br>
> [IMAG Lab](https://imag-njust.net/), Nanjing University of Science and Technology

> If STCDiT is helpful for you, please help star the GitHub Repo. Thanks!

> Welcome to visit our website (专注底层视觉领域的信息服务平台) for low-level vision: [https://lowlevelcv.com/](https://lowlevelcv.com/)
---


😊 You may also want to check our relevant works:

1. **STCDiT (CVPR 2026)** [Paper](https://arxiv.org/abs/2511.18786) | [Code](https://github.com/JyChen9811/STCDiT)  
   A motion-aware VAE and anchor-frame-guided DiT framework enables stable video restoration, even under complex camera motions.
   
2. **CODSR (CVPR2026)** [Paper](https://arxiv.org/abs/2512.14061  ) | [Code](https://github.com/Chanson94/CODSR  )  
   A one-step diffusion SR framework enabling region-discriminative activation of generative priors and precise semantic grounding.

### 🚩 **New Features/Updates**
- ✅ April 15, 2026. Release [enhanced results](https://modelscope.cn/datasets/junyangchen/STCDiT_large_results) of STCDiT on VideoLQ and SportsLQ.
- ✅ April 15, 2026. Release [SportsLQ](https://modelscope.cn/datasets/junyangchen/SportsLQ). It includes 20 sports event videos at 720p resolution.
- ✅ April 15, 2026. Release testing code and [pre-trained model](https://modelscope.cn/models/junyangchen/STCDiT_ckpt).
- ✅ November 24, 2025. Create the repository.

### ⚡ **To do**
- Release the training code. Note that STCDiT-tiny can be trained on 4×24 GB GPUs with the same training settings as in paper.
- Release the Gradio Demo and ComfyUI Integration.
- ~~Release the testing code and pre-trained model. Note that STCDiT-tiny can be inferred on a single 24 GB GPU.~~


### 📷 Real-World Enhancement Results



<table>
  <tr>
    <td>
      <video controls width="320" height="180"
             src="https://github.com/user-attachments/assets/71b04a2d-08f1-410c-8a00-f92bff6f8515">
      </video>
    </td>
    <td>
      <video controls width="320" height="180"
             src="https://github.com/user-attachments/assets/e3555158-27bf-4b42-8fe1-9a9c1f29a9bf">
      </video>
    </td>
  </tr>
  <tr>
    <td>
      <video controls width="320" height="180"
             src="https://github.com/user-attachments/assets/d307af1b-5f2b-4f4b-b14c-73002e0cd144">
      </video>
    </td>
    <td>
      <video controls width="320" height="180"
             src="https://github.com/user-attachments/assets/1c6c01f7-233a-4116-9fc7-84e3b3086250">
      </video>
    </td>
  </tr>
  <tr>
    <td>
      <video controls width="320" height="180"
             src="https://github.com/user-attachments/assets/0304b95e-fa5c-4639-9838-736d52189736">
      </video>
    </td>
    <td>
      <video controls width="320" height="180"
             src="https://github.com/user-attachments/assets/45425947-1d5b-4cd2-8404-74ca83475560">
      </video>
    </td>
  </tr>
</table>

---

### 🚀 How to evaluate


#### Environment
```shell
conda create -n STCDiT python=3.10.19 -y
pip install -r ./requirements_for_STCDiT.txt

conda create -n Qwen python=3.10.19 -y
pip install -r ./requirements_for_Qwen.txt

```
> **Note:** If FlashAttention installation fails, download the [.whl file](https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.2.post1/flash_attn-2.7.2.post1+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl)  and install it via pip.

#### Download Dependent Models
- [STCDiT and STCDiT-Tiny](https://modelscope.cn/models/junyangchen/STCDiT_ckpt)
- [Wan2.1-i2v-14B](https://www.modelscope.cn/models/Wan-AI/Wan2.1-I2V-14B-720P)
- [Wan2.1-t2v-1.3B](https://www.modelscope.cn/models/Wan-AI/Wan2.1-T2V-1.3B)
- [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)
- Put them in the `./model_checkpoints` folder. For download instructions, refer to download.sh.


#### Val Dataset
- SportsLQ: [Modelscope](https://modelscope.cn/datasets/junyangchen/SportsLQ)
- Enhanced results of STCDiT on VideoLQ and SportsLQ: [Modelscope](https://modelscope.cn/datasets/junyangchen/STCDiT_large_results).
- For download instructions, refer to download.sh.


#### Inference Script

> **Note:** Please modify line 3 in `./Inference/test_STCDiT_large.py` and `./Inference/test_STCDiT_tiny.py` to your local directory path.

```Shell
# Step 1: Generate Captions with Qwen2.5-VL
conda activate Qwen
bash ./Qwen2.5-VL/inference.sh

# Step 2: Run Video Super-Resolution with STCDiT
conda activate STCDiT

# STCDiT-Large with Wan2.1-I2V-14B base model, if you observe frequent texture flickering, set `cfg_scale=1`.
bash ./Inference/test_STCDiT_large.sh

# STCDiT-Tiny with Wan2.1-T2V-1.3B base model (a single 24 GB GPU is sufficient)
bash ./Inference/test_STCDiT_tiny.sh
```


### BibTeX
    @inproceedings{chen_STCDiT,
    title={STCDiT: Spatio-Temporally Consistent Diffusion Transformer for High-Quality Video Super-Resolution},
    author={Chen, Junyang and Jiangxin Dong and Long Sun and Yixin Yang and Pan, Jinshan},
    booktitle={CVPR},
    year={2026}
    }
---





### Contact
If you have any questions, please feel free to reach me out at `jychen9811@gmail.com`.

---

### Acknowledgments
Our project is based on [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) and [Wan 2.1](https://github.com/Wan-Video/Wan2.1). Thanks for their awesome works.






