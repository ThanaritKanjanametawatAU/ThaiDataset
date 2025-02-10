import torch
import torchaudio
from datasets import load_dataset, Dataset, Audio
from resemble_enhance.enhancer.inference import denoise, enhance
import numpy as np
import soundfile as sf
import gc

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
    # Immediately move output to CPU and clear CUDA cache
    output = wav2.cpu().numpy()
    if device == "cuda":
        torch.cuda.empty_cache()
    return (new_sr, output)

def process_batch(items, device, batch_size=500):
    """Process audio items in batches with memory management"""
    processed_items = []
    for i, item in enumerate(items, 1):
        print(f"Processing item {i}")
        
        new_sr, enhanced_audio = process_audio(item["audio"], device)
        enhanced_normalized = normalize_audio(enhanced_audio)
        
        processed_items.append({
            "sentence": item["sentence"],
            "audio": {"array": enhanced_normalized, "sampling_rate": new_sr}
        })
        
        # Clear memory after every batch_size items
        if i % batch_size == 0:
            print(f"Clearing memory after batch {i // batch_size}")
            if device == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
            
            # Optional: Print memory stats for monitoring
            if device == "cuda":
                print(f"GPU Memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
                print(f"GPU Memory cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
    
    return processed_items

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = load_dataset("Porameht/processed-voice-th-169k")
    batch_size = 500  # Process 500 items before clearing memory
    
    for split in ["train", "dev", "test"]:
        if split not in dataset:
            continue
            
        split_name = "validation" if split == "dev" else split
        print(f"Processing {split} split")
        
        # Process the entire split in batches
        processed_items = process_batch(dataset[split], device, batch_size)
        
        # Create and push dataset
        processed_dataset = Dataset.from_list(processed_items)
        processed_dataset = processed_dataset.cast_column("audio", Audio(sampling_rate=processed_items[0]["audio"]["sampling_rate"]))
        processed_dataset.push_to_hub(f"Thanarit/TH-Speech-Enhanced", split=split_name)
        
        # Clear memory after each split
        if device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    main()