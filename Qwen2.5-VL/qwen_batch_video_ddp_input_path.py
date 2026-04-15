
import os
import json
from glob import glob
from tqdm import tqdm
import argparse 

import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import random

MODEL_PATH = "./model_checkpoints/Qwen-VL"
DTYPE = torch.bfloat16
MAX_NEW_TOKENS = 256
# =================================

class VideoDataset(Dataset):
    def __init__(self, video_file_path, json_folder): 

        self.json_folder = json_folder
        nature_paths = sorted(glob(os.path.join(video_file_path, '**', '*.mp4'), recursive=True)) + sorted(glob(os.path.join(video_file_path, '**', '*.MOV'), recursive=True))
        self.filtered_list = nature_paths
        self.video_file_path = video_file_path # 存储路径以便在 __getitem__ 中使用

    def __len__(self):
        return len(self.filtered_list)

    def __getitem__(self, idx):
        video_path = self.filtered_list[idx]
        # Prepare the message
        
        message = {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "max_pixels": 240 * 240,
                    "fps": 1.0,
                },
                {"type": "text", "text": "Describe this video. Pay attention to all objects in the video. The description should be useful for AI to re-generate the video. The description should be no more than six sentences. Here are some examples of good descriptions: 1. A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage. She wears a black leather jacket, a long red dress, and black boots, and carries a black purse. She wears sunglasses and red lipstick. She walks confidently and casually. The street is damp and reflective, creating a mirror effect of the colorful lights. Many pedestrians walk about. 2. Several giant wooly mammoths approach treading through a snowy meadow, their long wooly fur lightly blows in the wind as they walk, snow covered trees and dramatic snow capped mountains in the distance, mid afternoon light with wispy clouds and a sun high in the distance creates a warm glow, the low camera view is stunning capturing the large furry mammal with beautiful photography, depth of field. 3. Drone view of waves crashing against the rugged cliffs along Big Sur's garay point beach. The crashing blue waters create white-tipped waves, while the golden light of the setting sun illuminates the rocky shore. A small island with a lighthouse sits in the distance, and green shrubbery covers the cliff's edge. The steep drop from the road down to the beach is a dramatic feat, with the cliff’s edges jutting out over the sea. This is a view that captures the raw beauty of the coast and the rugged landscape of the Pacific Coast Highway.",},
            ],
            }
        

        json_path = os.path.basename(video_path).replace('.mp4', '.json')
        json_path = os.path.join(self.json_folder, json_path)
        
        return {'message': [message], 'json_name': json_path}

def setup_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl", init_method="env://")
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        is_distributed = True
    else:
        rank, world_size, local_rank = 0, 1, 0
        is_distributed = False
    return is_distributed, rank, world_size, local_rank


def cleanup_distributed(is_distributed):
    if is_distributed and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()




def load_model_and_processor(local_rank):
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=DTYPE,
        attn_implementation="flash_attention_2",
        device_map=None, 
    ).to(device)


    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    processor.tokenizer.padding_side = "left"
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    return model, processor, device


def main():
    parser = argparse.ArgumentParser(description="Multi-GPU Qwen-VL Inference for Image Captioning")
    parser.add_argument(
        "--video_file_path",
        type=str,
        required=True, 
        help="The root directory containing video files (e.g., /path/to/HR_crops)."
    )

    parser.add_argument(
        "--json_folder",
        type=str,
        required=True,
        help="The root directory containing json files."
    )

    args = parser.parse_args()
    # ----------------------

    os.makedirs(args.json_folder, exist_ok=True)
    is_distributed, rank, world_size, local_rank = setup_distributed()
    try:

        def collate_fn(examples):
        
            message = [example["message"] for example in examples]
            json_name = [example["json_name"] for example in examples]
            return {"message": message, "json_name": json_name}

        torch.backends.cuda.matmul.allow_tf32 = True
        model, processor, device = load_model_and_processor(local_rank)
        

        dataset = VideoDataset(video_file_path=args.video_file_path, json_folder=args.json_folder)
        inference_sampler = DistributedSampler(dataset)
        train_dataloader = torch.utils.data.DataLoader(
            dataset,
            sampler=inference_sampler,
            shuffle=False,
            collate_fn=collate_fn,
            batch_size=1,
            num_workers=4,
        )


        progress_bar = tqdm(
            range(0, len(train_dataloader)),
            initial=0,
            desc="Steps",
            disable=(rank != 0)
        )         


        for step, batch in enumerate(train_dataloader):

            messages = batch["message"]
            texts = [
                    processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
                    for msg in messages
                ]

            image_input_list = []
            video_inputs_list = []
            for msg in messages:
                _image_inputs, _video_inputs = process_vision_info(msg)
                image_input_list.append(_image_inputs)
                video_inputs_list.append(_video_inputs[0])

            inputs = processor(
                    text=texts,
                    videos=video_inputs_list,
                    padding=True,
                    return_tensors="pt",
                )

            inputs = inputs.to(device)

            
            with torch.inference_mode():
                generated_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
                generated_ids_trimmed = [
                    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_texts = processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )

            for j in range(len(output_texts)):
                cap_j = output_texts[j].replace('\n','')

                file_path = batch['json_name'][j]
                data = {
                    "caption": cap_j
                }

                with open(file_path, "w") as json_file:
                    json.dump(data, json_file, indent=4)
                    
            progress_bar.update(1)


        if is_distributed:
            dist.barrier()

    finally:
        cleanup_distributed(is_distributed)


if __name__ == "__main__":
    main()