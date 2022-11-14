import argparse
import random
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default="./filelists/example_audio_filelist.txt", help='filelist path')
    parser.add_argument('-o', '--output', type=str, default="./filelists", help='File list output path')
    args = parser.parse_args()

    random.seed(1234)

    with open(args.input, "r", encoding="utf-8") as f:
        lines = f.readlines()

    random.shuffle(lines)

    origin_filename = os.path.basename(args.input)
    data_len = len(lines)

    with open(os.path.join(args.output, origin_filename.replace(".txt", "_train.txt")), "w", encoding="utf-8") as f:
        f.writelines(lines[:-20])
    
    with open(os.path.join(args.output, origin_filename.replace(".txt", "_valid.txt")), "w", encoding="utf-8") as f:
        f.writelines(lines[-20:-10])
    
    with open(os.path.join(args.output, origin_filename.replace(".txt", "_test.txt")), "w", encoding="utf-8") as f:
        f.writelines(lines[-10:])