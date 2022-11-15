import argparse
import glob
import os
import tqdm
import soundfile as sf

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default="./dataset", help='Dataset path')
    parser.add_argument('-o', '--output', type=str, default="./filelists/example_audio_filelist.txt", help='File list output path')
    args = parser.parse_args()

    audio_files = list(glob.glob(os.path.join(args.input, "**/*.wav"), recursive=True))

    total_time = 0
    with open(args.output, "w", encoding="utf-8") as f:
        for i, audio_path in enumerate(tqdm.tqdm(audio_files)):
            audio = sf.SoundFile(audio_path)
            sec = audio.frames / audio.samplerate
            total_time += sec
            # if audio.frames / audio.samplerate < 1:
            #     continue
            audio_path = audio_path.replace("\\", "/")
            f.write(f"{audio_path}\n")
    
    print(f"Total time: {total_time//3600}h")