import torch
import torchaudio
from datasets import load_dataset, Dataset, Audio
from resemble_enhance.enhancer.inference import denoise, enhance
import numpy as np
import soundfile as sf

def normalize_audio(audio_array):
    return audio_array / np.max(np.abs(audio_array))

@torch.inference_mode()
def process_audio(audio_dict, device):
    audio_array = audio_dict['array']
    sr = audio_dict['sampling_rate']
    dwav = torch.from_numpy(audio_array).float()
    if len(dwav.shape) > 1:
        dwav = dwav.mean(dim=0)
    wav2, new_sr = enhance(dwav, sr, device, nfe=80, solver="midpoint", lambd=0.1, tau=0.5)
    return (new_sr, wav2.cpu().numpy())



def main():
   device = "cuda" if torch.cuda.is_available() else "cpu"
   dataset = load_dataset("Porameht/processed-voice-th-169k")
   
   for split in ["train", "dev", "test"]:
       if split not in dataset:
           continue
           
       split_name = "validation" if split == "dev" else split
       processed_items = []
       i = 1
       
       for item in dataset[split]:
           print(f"Processing {split} {i}")
           i += 1

           new_sr, enhanced_audio = process_audio(item["audio"], device)
           enhanced_normalized = normalize_audio(enhanced_audio)
           
           processed_items.append({
               "sentence": item["sentence"],
               "audio": {"array": enhanced_normalized, "sampling_rate": new_sr}
           })
           

       
       processed_dataset = Dataset.from_list(processed_items)
       processed_dataset = processed_dataset.cast_column("audio", Audio(sampling_rate=new_sr))
       processed_dataset.push_to_hub(f"Thanarit/TH-Speech-Enhanced", split=split_name)

if __name__ == "__main__":
   main()