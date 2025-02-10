class TextCleaner:
    def __init__(self):
        """Initialize text cleaner with Thai phoneme dictionary"""
        self.word_index_dictionary = self._initialize_phoneme_dict()
    
    def _initialize_phoneme_dict(self) -> Dict[str, int]:
        """Create a mapping of Thai phonemes to indices"""
        # This is a placeholder - you'll need to define your actual phoneme set
        phonemes = ['_', ' '] + list('กขคฆงจฉชซฌญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรลวศษสหฬอฮะาิีึืุูเแโใไๅํ็่้๊๋์')
        return {p: i for i, p in enumerate(phonemes)}
    
    def __call__(self, text: str) -> str:
        """Convert Thai text to phonemes"""
        # This is a simplified version - you'll need to implement proper Thai text to phoneme conversion
        return ' '.join(text)  # Placeholder implementation

class DatasetPreprocessor:
    def __init__(self, target_sr=24000):
        self.audio_processor = EnhancedAudioPreprocessor(target_sr=target_sr)
        self.text_cleaner = TextCleaner()
        
    def prepare_for_training(self, dataset):
        """
        Transform the dataset into the format expected by the training pipeline
        """
        def process_example(example):
            # 1. Process audio as before
            processed_audio = self.audio_processor.process_and_verify(
                example['audio']['array'],
                example['audio']['sampling_rate']
            )
            
            # 2. Convert text to phonemes
            text = example['sentence']
            phonemes = self.text_cleaner(text)
            
            # 3. Create mel spectrogram
            mel_tensor = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.audio_processor.target_sr,
                n_fft=2048,
                hop_length=300,  # Matching the model's expectations
                n_mels=80
            )(torch.FloatTensor(processed_audio))
            
            # 4. Structure the example in the required format
            return {
                'audio_path': example['path'] if 'path' in example else '',  # Handle missing path
                'text': phonemes,  # Phonemized text
                'mel': mel_tensor.numpy(),
                'speaker_id': 0,  # Adding default speaker ID
                'input_lengths': len(phonemes),
                'mel_lengths': mel_tensor.shape[1]
            }

        # Transform dataset
        training_dataset = dataset.map(
            process_example,
            remove_columns=dataset.column_names,
            num_proc=4
        )
        
        return training_dataset

def save_training_lists(dataset, output_dir: str):
    """Save dataset splits in the format expected by the training pipeline"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create the data list files expected by the model
    splits = ['train', 'dev', 'test']
    for split in splits:
        output_file = os.path.join(output_dir, f'{split}_list.txt')
        with open(output_file, 'w', encodingc='utf-8') as f:
            for item in dataset[split]:
                f.write(f"{item['audio_path']}|{item['text']}|{item['speaker_id']}\n")

def prepare_dataset_for_training():
    """
    Prepare the cleaned dataset for the training pipeline
    """
    # Load the cleaned dataset
    dataset = load_dataset("Thanarit/TH-Speech-Cleaned")
    
    # Initialize preprocessor
    preprocessor = DatasetPreprocessor()
    
    # Process dataset into training format
    training_dataset = preprocessor.prepare_for_training(dataset)
    
    # Save in the format expected by the training pipeline
    data_dir = "Data"
    save_training_lists(training_dataset, data_dir)
    
    # Create config file for training
    config = {
        'data_params': {
            'train_data': 'Data/train_list.txt',
            'val_data': 'Data/dev_list.txt',
            'test_data': 'Data/test_list.txt',
            'root_path': 'Data',
            'min_length': 50,  # Minimum sequence length
        },
        'model_params': {
            'n_token': len(preprocessor.text_cleaner.word_index_dictionary),
            'hidden_dim': 512,
            'n_layer': 6,
            'multispeaker': False,  # Single speaker for now
        }
    }
    
    with open('config_thai.yml', 'w') as f:
        yaml.dump(config, f)
    
    return training_dataset, config


def test_preprocessing():
    """
    Test the preprocessing on a small subset and display results
    """
    dataset = load_dataset("Porameht/processed-voice-th-169k")
    preprocessor = AudioPreprocessor()
    
    # Process a few examples
    test_examples = dataset['train'].select(range(5))
    
    for idx, example in enumerate(test_examples):
        print(f"\nProcessing example {idx + 1}")
        
        # Get original audio
        waveform = example['audio']['array']
        sr = example['audio']['sampling_rate']
        
        # Process audio
        processed = preprocessor.process_and_verify(waveform, sr, visualize=True)
        
        # Play original and processed audio
        print(f"Text: {example['sentence']}")
        print("\nOriginal Audio:")
        display(IPython.display.Audio(waveform, rate=sr))
        print("\nProcessed Audio:")
        display(IPython.display.Audio(processed, rate=preprocessor.target_sr))