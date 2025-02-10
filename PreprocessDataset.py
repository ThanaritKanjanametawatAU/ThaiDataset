import datasets
from datasets import load_dataset, Audio, Dataset
import librosa
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchaudio
import noisereduce as nr
from scipy import signal
from huggingface_hub import HfApi
import os
import warnings
import yaml
from typing import Dict, Any
import scipy.ndimage
import scipy.signal as signal
import pyloudnorm as pyln
from pedalboard import Pedalboard, Compressor, Gain, HighpassFilter, LowpassFilter, Reverb
import traceback
import gc
import itertools
import traceback
from memory_profiler import profile  # Add this to help monitor memory usage
from resemble_enhance.enhancer.inference import denoise, enhance

warnings.filterwarnings('ignore')

class AudioPreprocessor:
    def __init__(self, target_sr=24000):
        self.target_sr = target_sr
        self.min_audio_length = 2048  # Minimum length needed for processing
        
        # We'll use more conservative parameters focused on speech preservation
        self.noise_params = {
            'prop_decrease': 0.70,    # Reduced from 0.85 to be more gentle
            'n_std_thresh': 1.5,      # Standard deviation threshold
            'n_fft': 2048,            # For better frequency resolution
            'win_length': 1024,       # Shorter window for better time resolution
            'hop_length': 256         # Small hop length for smoother results
        }

    def enhance_audio(self, waveform, sr, device="cuda"):
        """Apply resemble-enhance processing first"""
        try:
            # Convert to torch tensor
            dwav = torch.from_numpy(waveform).float()
            if len(dwav.shape) > 1:
                dwav = dwav.mean(dim=0)
                
            # Apply enhancement
            with torch.inference_mode():
                wav2, new_sr = enhance(dwav, sr, device, nfe=80, solver="midpoint", lambd=0.1, tau=0.5)
                enhanced = wav2.cpu().numpy()
            
            # # Normalize
            # enhanced = enhanced / np.max(np.abs(enhanced))
            
            # Resample if needed
            # if new_sr != self.target_sr:
            #     enhanced = librosa.resample(enhanced, orig_sr=new_sr, target_sr=self.target_sr)
                
            return enhanced
            
        except Exception as e:
            print(f"Warning: Enhancement failed with error: {str(e)}. Using original audio.")
            return waveform

    def clean_audio(self, waveform, sr):
        """
        Enhanced cleaning approach that first applies resemble-enhance
        """
        try:    
            # cleaning steps
            if len(waveform) < self.min_audio_length:
                pad_length = self.min_audio_length - len(waveform)
                waveform = np.pad(waveform, (0, pad_length), mode='reflect')
            
            # First ensure we're working with the right sample rate
            if sr != self.target_sr:
                waveform = librosa.resample(waveform, orig_sr=sr, target_sr=self.target_sr)

            # Remove any DC offset which can cause issues
            waveform = waveform - np.mean(waveform)

            try:
                # Apply gentle noise reduction
                cleaned = nr.reduce_noise(
                    y=waveform,
                    sr=self.target_sr,
                    stationary=True,  # Using stationary mode for more stable results
                    prop_decrease=self.noise_params['prop_decrease'],
                    n_std_thresh_stationary=self.noise_params['n_std_thresh']
                )
            except ValueError as e:
                print(f"Warning: Noise reduction failed, using original audio. Error: {str(e)}")
                cleaned = waveform

            # Normalize the audio to a reasonable level while preserving dynamics
            peak = np.abs(cleaned).max()
            if peak > 0:
                cleaned = cleaned * (0.7 / peak)  # Leaving headroom to prevent clipping

            # Apply very gentle noise gate to remove any remaining low-level noise
            noise_gate_threshold = 0.001  # Very low threshold to preserve speech
            gate_mask = np.abs(cleaned) > noise_gate_threshold
            cleaned = cleaned * gate_mask

            try:
                # Trim silence from ends but keep a small padding
                cleaned, _ = librosa.effects.trim(
                    cleaned,
                    top_db=20,         # Less aggressive trimming
                    frame_length=1024,
                    hop_length=256
                )
            except ValueError as e:
                print(f"Warning: Trimming failed, using untrimmed audio. Error: {str(e)}")

            # Add small padding for natural sound
            pad_ms = 100  # 50ms padding
            pad_length = int(self.target_sr * pad_ms / 1000)
            cleaned = np.pad(cleaned, (pad_length, pad_length), mode='constant')

                        # First apply enhancement
            device = "cuda" if torch.cuda.is_available() else "cpu"
            cleaned = self.enhance_audio(cleaned, sr, device)

            return cleaned

        except Exception as e:
            print(f"Warning: Audio processing failed with error: {str(e)}. Returning normalized original audio.")
            # Return normalized original audio as fallback
            if len(waveform) == 0:
                return np.zeros(self.min_audio_length)
            return librosa.util.normalize(waveform)

    def process_and_verify(self, waveform, sr):
        """
        Process audio and verify that we haven't made it worse
        """
        try:
            # Check for empty or invalid audio
            if len(waveform) == 0 or not np.any(waveform):
                print("Warning: Empty or invalid audio detected")
                return np.zeros(self.min_audio_length)

            # Process the audio
            processed = self.clean_audio(waveform, sr)
            
            # Calculate quality metrics for before and after
            def get_metrics(audio):
                return {
                    'rms': np.sqrt(np.mean(audio**2)),
                    'peak': np.abs(audio).max(),
                    'dynamic_range': np.percentile(np.abs(audio), 95) / (np.percentile(np.abs(audio), 5) + 1e-8)
                }
            
            original_metrics = get_metrics(waveform)
            processed_metrics = get_metrics(processed)
            
            # If processing made the audio significantly worse, return the original
            if (processed_metrics['rms'] < original_metrics['rms'] * 0.5 or
                processed_metrics['dynamic_range'] < original_metrics['dynamic_range'] * 0.5):
                print("Warning: Processing may have degraded audio quality. Using original with basic normalization.")
                return librosa.util.normalize(waveform)
                
            return processed

        except Exception as e:
            print(f"Warning: Audio verification failed with error: {str(e)}. Returning normalized original audio.")
            # Return normalized original audio as fallback
            if len(waveform) == 0:
                return np.zeros(self.min_audio_length)
            return librosa.util.normalize(waveform)
    



class AudioQualityAuditor:
    def __init__(self, output_dir="audit_results"):
        """
        Initialize auditor with output directory for saving results
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize metrics tracking
        self.metrics_log = {
            'snr': [],           # Signal-to-noise ratio
            'peak_amplitude': [], # Peak amplitude
            'duration': [],      # Duration in seconds
            'silence_ratio': [], # Ratio of silence to speech
            'clipping': [],      # Clipping detection
            'rms': [],          # Root mean square energy
            'zero_crossings_rate': [], # Zero crossings rate
            'spectral_centroid': [], # Spectral centroid
            'spectral_rolloff': [], # Spectral rolloff
            'issues': []         # Track problematic samples
        }

    def calculate_audio_metrics(self, waveform, sr):
        """
        Calculate comprehensive audio quality metrics
        """
        # Get basic metrics
        duration = len(waveform) / sr
        
        # Calculate signal energy
        signal_energy = np.mean(waveform ** 2)
        
        # Estimate noise floor from silent regions
        silent_regions = waveform[waveform ** 2 < np.percentile(waveform ** 2, 10)]
        noise_energy = np.mean(silent_regions ** 2) if len(silent_regions) > 0 else 1e-10
        
        # Calculate SNR
        snr = 10 * np.log10(signal_energy / noise_energy) if noise_energy > 0 else 100
        
        # Calculate silence ratio
        silence_threshold = 0.01
        silence_ratio = np.mean(np.abs(waveform) < silence_threshold)
        
        # Check for clipping
        clipping_ratio = np.mean(np.abs(waveform) > 0.99)
        
        # Get detailed metrics
        detailed_metrics = self.calculate_detailed_metrics(waveform, sr)
        
        return {
            'snr': snr,
            'peak_amplitude': np.max(np.abs(waveform)),
            'duration': duration,
            'silence_ratio': silence_ratio,
            'clipping': clipping_ratio,
            **detailed_metrics  # Include all detailed metrics
        }

    def calculate_detailed_metrics(self, waveform, sr):
        """Calculate comprehensive audio quality metrics"""
        return {
            'rms': np.sqrt(np.mean(waveform**2)),
            'peak': np.abs(waveform).max(),
            'zero_crossings_rate': librosa.feature.zero_crossing_rate(waveform)[0].mean(),
            'spectral_centroid': librosa.feature.spectral_centroid(y=waveform, sr=sr)[0].mean(),
            'spectral_rolloff': librosa.feature.spectral_rolloff(y=waveform, sr=sr)[0].mean()
        }

    def verify_audio_quality(self, metrics, threshold_config={
        'min_rms': 0.01,
        'max_zero_crossing_rate': 0.3,
        'min_spectral_centroid': 500,
        'max_spectral_centroid': 4000
    }):
        """Verify that processed audio meets quality standards"""
        issues = []
        
        if metrics['rms'] < threshold_config['min_rms']:
            issues.append('too_quiet')
        if metrics['zero_crossings_rate'] > threshold_config['max_zero_crossing_rate']:
            issues.append('possible_noise')
        if not (threshold_config['min_spectral_centroid'] <= metrics['spectral_centroid'] <= threshold_config['max_spectral_centroid']):
            issues.append('frequency_content_issue')
        
        return issues

    def save_sample_audio(self, waveform, sr, index, prefix="sample"):
        """
        Save audio sample for later review
        """
        filepath = os.path.join(self.output_dir, f"{prefix}_{index}.wav")
        sf.write(filepath, waveform, sr)
        return filepath
    

def get_noise_floor(signal):
        percentile = np.percentile(np.abs(signal), 10)
        noise = signal[np.abs(signal) < percentile]
        return np.mean(noise**2) if len(noise) > 0 else 1e-10


def calculate_snr_improvement(original, processed):
    """
    Calculate SNR improvement in a way that's more relevant to speech
    """


    original_noise = get_noise_floor(original)
    processed_noise = get_noise_floor(processed)
    
    original_signal = np.mean(original**2)
    processed_signal = np.mean(processed**2)
    
    original_snr = 10 * np.log10(original_signal / original_noise)
    processed_snr = 10 * np.log10(processed_signal / processed_noise)
    
    return processed_snr - original_snr


def process_batch(batch_idx, examples, preprocessor, output_dir=None, is_audit=False, save_interval=100):
    """
    A unified batch processing function that handles both audit and dataset processing cases.
    
    Args:
        batch_idx (int): Index of the current batch
        examples (list): List of examples to process
        preprocessor (AudioPreprocessor): Instance of AudioPreprocessor
        output_dir (str, optional): Directory to save processed files
        is_audit (bool): Whether this is an audit run or full processing
        save_interval (int): Interval for saving sample files
    
    Returns:
        list: Processed examples with metrics if auditing, or processed examples if not
    """
    processed_results = []
    examples = list(examples)  # Convert to list for consistent iteration
    
    for idx, example in enumerate(examples):
        try:
            # Get audio data with proper error handling
            audio_data = example.get('audio', {})
            if not audio_data or 'array' not in audio_data:
                print(f"Missing audio data in example {batch_idx * len(examples) + idx}")
                continue
                
            # Convert to float32 immediately to save memory
            original = audio_data['array'].astype(np.float32)
            original_sr = audio_data['sampling_rate']
            
            # Calculate global index for tracking
            global_idx = batch_idx * len(examples) + idx
            
            # Calculate SNR metrics before processing if this is an audit
            if is_audit:
                # Store original metrics before modifying the audio
                original_signal = np.mean(original**2)
                original_noise = get_noise_floor(original)
                original_snr = 10 * np.log10(original_signal / original_noise) if original_noise > 0 else 100
            
            # Process the audio
            processed = preprocessor.process_and_verify(original, original_sr)
            
            # Save periodic samples if requested
            if output_dir and global_idx % save_interval == 0:
                sample_dir = os.path.join(output_dir, 'audit_samples' if is_audit else 'verification_samples')
                os.makedirs(sample_dir, exist_ok=True)
                
                # Save both original and processed samples
                sf.write(
                    os.path.join(sample_dir, f'original_{global_idx}.wav'),
                    original,
                    original_sr
                )
                sf.write(
                    os.path.join(sample_dir, f'processed_{global_idx}.wav'),
                    processed,
                    preprocessor.target_sr
                )
            
            if is_audit:
                # Calculate SNR improvement using the stored original metrics
                processed_signal = np.mean(processed**2)
                processed_noise = get_noise_floor(processed)
                processed_snr = 10 * np.log10(processed_signal / processed_noise) if processed_noise > 0 else 100
                snr_improvement = processed_snr - original_snr
                
                # Store audit metrics
                processed_results.append({
                    'snr_improvement': snr_improvement,
                    'text': example.get('sentence', ''),
                    'global_idx': global_idx
                })
            else:
                # For regular processing, store processed audio
                processed_results.append({
                    'sentence': example.get('sentence', ''),
                    'audio': {
                        'array': processed,
                        'sampling_rate': preprocessor.target_sr
                    },
                })
            
            # Clean up memory
            del original
            del processed
            
            # Periodic garbage collection
            if idx % 300 == 0:
                gc.collect()
                
        except Exception as e:
            print(f"Error processing example {batch_idx * len(examples) + idx}: {str(e)}")
            traceback.print_exc()
            # Handle errors appropriately based on mode
            if not is_audit:
                processed_results.append(example)
            else:
                processed_results.append({
                    'snr_improvement': None,
                    'text': example.get('sentence', ''),
                    'global_idx': global_idx,
                    'error': str(e)
                })
    
    # Final garbage collection for the batch
    gc.collect()
    return processed_results

def audit_dataset_processing(num_samples=1000, save_interval=100):
    """
    Audit the dataset processing with quality checks using the unified process_batch function.
    """
    preprocessor = AudioPreprocessor()
    output_dir = "audit_results"
    os.makedirs(output_dir, exist_ok=True)
    
    dataset = load_dataset("Thanarit/TH-Speech-Enhanced")
    
    # Process in batches
    batch_size = 1
    total_batches = num_samples // batch_size
    
    all_metrics = []
    for batch_idx in range(total_batches):
        print(f"Processing batch {batch_idx + 1}/{total_batches}...")
        
        batch = dataset['train'].select(range(batch_idx * batch_size, (batch_idx + 1) * batch_size))
        batch_metrics = process_batch(
            batch_idx=batch_idx,
            examples=batch,
            preprocessor=preprocessor,
            output_dir=output_dir,
            is_audit=True,
            save_interval=save_interval
        )
        all_metrics.extend(batch_metrics)
        
        # Calculate and display batch statistics
        valid_metrics = [m['snr_improvement'] for m in batch_metrics if m['snr_improvement'] is not None]
        if valid_metrics:
            avg_snr = np.mean(valid_metrics)
            print(f"Batch {batch_idx + 1} average SNR improvement: {avg_snr:.2f} dB")
        
    return all_metrics

def process_and_push_dataset():
    """
    Process the dataset and push to Hugging Face with complete splits.
    Accumulates all processed examples before pushing each split in its entirety.
    """
    preprocessor = AudioPreprocessor()
    output_dir = "processed_dataset"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset with streaming enabled for memory efficiency
    dataset = load_dataset(
        "Porameht/processed-voice-th-169k",
        streaming=True
    )
    
    batch_size = 100 
    api = HfApi()
    repo_id = "Thanarit/TH-Speech-Cleaned"
    
    # Create a temporary directory for audio files
    temp_dir = "temp_audio_files"
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        api.create_repo(repo_id=repo_id, exist_ok=True)
        
        for split in ['train', 'dev', 'test']:
            print(f"\nProcessing {split} split...")
            split_data = dataset[split]
            
            # Initialize list to store all processed examples for this split
            all_processed_examples = []
            batch_idx = 0
            
            # Create iterator for streaming dataset
            split_iterator = iter(split_data)
            
            while True:
                try:
                    # Get next batch
                    batch = list(itertools.islice(split_iterator, batch_size))
                    if not batch:
                        break
                        
                    print(f"Processing batch {batch_idx + 1}...")
                    
                    # Process batch
                    processed_batch = process_batch(
                        batch_idx=batch_idx,
                        examples=batch,
                        preprocessor=preprocessor,
                        output_dir=output_dir,
                        is_audit=False,
                        save_interval=100
                    )
                    
                    # Convert and save processed examples
                    formatted_examples = []
                    for idx, example in enumerate(processed_batch):
                        try:
                            # Create a unique filename for each audio file
                            audio_path = os.path.join(temp_dir, f"{split}_{batch_idx}_{idx}.wav")
                            
                            # Save the audio file
                            sf.write(
                                audio_path,
                                example['audio']['array'],
                                example['audio']['sampling_rate']
                            )
                            
                            # Create formatted example
                            formatted_example = {
                                'audio': audio_path,
                                'sentence': example['sentence'],
                            }
                            formatted_examples.append(formatted_example)
                            
                        except Exception as e:
                            print(f"Error formatting example {idx}: {str(e)}")
                            continue
                    
                    # Add formatted examples to the complete list
                    all_processed_examples.extend(formatted_examples)
                    
                    batch_idx += 1
                    
                    # Perform garbage collection after each batch
                    gc.collect()
                    
                except Exception as e:
                    print(f"Error in batch {batch_idx}: {str(e)}")
                    traceback.print_exc()
                    continue
            
            # After processing all batches for this split, create and push the complete dataset
            print(f"Creating complete dataset for {split} split...")
            
            # Create complete dataset with Audio feature
            complete_dataset = Dataset.from_dict({
                k: [example[k] for example in all_processed_examples]
                for k in all_processed_examples[0].keys()
            })
            
            # Cast the audio column to Audio feature
            complete_dataset = complete_dataset.cast_column("audio", Audio())
            
            # Push the complete split
            print(f"Pushing complete {split} split to Hub...")
            complete_dataset.push_to_hub(
                repo_id,
                split=split,  # Use the original split name
                private=False
            )
            
            # Clean up temporary files for this split
            print(f"Cleaning up temporary files for {split} split...")
            for example in all_processed_examples:
                if os.path.exists(example['audio']):
                    os.remove(example['audio'])
            
            # Clear the list and force garbage collection
            all_processed_examples.clear()
            del complete_dataset
            gc.collect()
            
            print(f"Completed processing and pushing {split} split")
        
        # Final cleanup of temporary directory
        shutil.rmtree(temp_dir)
            
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return
    
    return "Dataset processing and pushing complete"


def main():
    """
    Review audit results and proceed with dataset processing if quality is acceptable
    """
    # First run the audit
    print("Starting dataset audit...")
    audit_stats = audit_dataset_processing(num_samples=9, save_interval=1) 
    
    # Print audit results
    print("\nAudit Results:")
    print(f"Average SNR improvement: {np.mean([m['snr_improvement'] for m in audit_stats]):.2f} dB")
    
    # Save sample files for manual review
    print("\nSample files have been saved to the audit_results directory")
    print("Please review them before proceeding with full dataset processing")
    
    # Ask for confirmation
    proceed = input("\nDo you want to proceed with processing the full dataset? (yes/no): ")
    
    if proceed.lower() == 'yes':
        print("Processing full dataset...")
        process_and_push_dataset()
    else:
        print("Dataset processing cancelled. Please review the audit results and adjust parameters as needed.")



if __name__ == "__main__":
    # Audit Review and Proceed
    main()