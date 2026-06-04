from pathlib import Path

src_dir = Path("./dataset/video")
out_file = Path("./dataset/video_list.txt")

mp4_files = sorted(src_dir.rglob("*.mp4"))

with out_file.open("w", encoding="utf-8") as f:
    for path in mp4_files:
        f.write(str(path) + "\n")

print(f"写入完成，共 {len(mp4_files)} 个 mp4 文件")
print(f"输出文件: {out_file}")