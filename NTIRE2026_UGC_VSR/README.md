### NTIRE 2026 ON SHORT-FORM UGC VIDEO RESTORATION IN THE WILD WITH GENERATIVE MODELS @ CVPR2026

> We directly applied the [pretrained STCDiT model](https://modelscope.cn/models/junyangchen/STCDiT_ckpt) **without any fine-tuning on the official video data**.

> We achieved **3rd place** in the User Study track against fine-tuned methods, demonstrating strong generalization. [[Challenge Report]](https://arxiv.org/abs/2604.10551)

| Team Name | Subjective Score | Ranking (User Study) |
|-----------|-----------------|---------------------|
| RedMediaTech | 3.8525 | 1 |
| TaoMC2 | 3.79875 | 2 |
| STCVSR(Ours) | 3.785 | 3 |
| MiAlgo LM | 3.781875 | 4 |

### 🚀 How to evaluate


#### Inference Script

```Shell
# Step 1: Decode videos to frames
python ./NTIRE2026_UGC_VSR/video_to_png.py

# Step 2: Apply SOTA Image SR to process frames in "./Video/LQ_images", save results to "./Video/enhanced_images"

# Step 3: Re-encode frames to video
python ./NTIRE2026_UGC_VSR/png_2_video_mix.py

# Step 4: Run STCDiT
bash ./Inference/test_STCDiT_large.sh
# or
bash ./Inference/test_STCDiT_tiny.sh
```
> **Tip:** For videos with extremely dense textures, we recommend directly using the LQ video as input to avoid flickering caused by inconsistent texture across frames in Step 2.







