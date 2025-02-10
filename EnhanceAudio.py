class EnhancedAudioPreprocessor(AudioPreprocessor):
    def __init__(self, target_sr=24000):
        super().__init__(target_sr)
        
        # Adjust parameters for better speech preservation
        self.target_lufs = -23.0  # Increased from -27 for better audibility
        self.peak_threshold = -1.0  # Increased from -3.0 for better volume
        
        # Modified processor parameters focused on speech
        self.processor_params = {
            'highpass_freq': 80,  # Increased to better preserve speech fundamentals
            'lowpass_freq': 12000,  # Reduced to focus on speech range
            'compressor': {
                'threshold_db': -24,  # Higher threshold for more natural dynamics
                'ratio': 2.0,  # Slightly stronger compression
                'attack_ms': 10,  # Faster attack to catch transients
                'release_ms': 100  # Faster release for more natural sound
            }
        }

    def _get_processor(self):
        """Create and return a new processor instance with tuned parameters"""
        return Pedalboard([
            HighpassFilter(cutoff_frequency_hz=self.processor_params['highpass_freq']),
            LowpassFilter(cutoff_frequency_hz=self.processor_params['lowpass_freq']),
            Compressor(
                threshold_db=self.processor_params['compressor']['threshold_db'],
                ratio=self.processor_params['compressor']['ratio'],
                attack_ms=self.processor_params['compressor']['attack_ms'],
                release_ms=self.processor_params['compressor']['release_ms']
            ),
            Gain(gain_db=6.0)  # Add some makeup gain
        ])

    def _normalize_loudness(self, waveform):
        """
        Enhanced normalization with better volume control and protection against outliers
        """
        # Initialize loudness meter
        meter = pyln.Meter(self.target_sr)
        
        # Measure current loudness
        current_loudness = meter.integrated_loudness(waveform)
        
        # Calculate peak values
        peak_loudness = np.max(np.abs(waveform))
        peak_db = 20 * np.log10(peak_loudness + 1e-8)
        
        # Calculate initial gain needed
        gain_db = self.target_lufs - current_loudness
        
        # Check if applying this gain would exceed our peak threshold
        resulting_peak = peak_db + gain_db
        if resulting_peak > self.peak_threshold:
            # If so, reduce the gain to respect peak threshold
            gain_db -= (resulting_peak - self.peak_threshold)
        
        # Apply adaptive ceiling based on content type
        if current_loudness > -15:  # If content is very loud
            # Apply additional reduction for very loud content
            gain_db -= (current_loudness + 15) * 0.5
        
        # Ensure we're not applying extreme gains
        gain_db = np.clip(gain_db, -20, 6)
        
        # Smoothly apply gain
        return self._apply_smooth_gain(waveform, gain_db)

    def _apply_smooth_gain(self, waveform, target_gain_db):
        """
        Apply gain changes smoothly to avoid abrupt volume changes
        """
        # Convert dB to linear gain
        target_gain = 10 ** (target_gain_db/20)
        
        # Create smoothing window (50ms)
        window_size = int(0.050 * self.target_sr)
        fade = np.hanning(window_size * 2)
        
        # Apply gain with smooth fade in/out
        gained = waveform * target_gain
        
        # Apply smooth fade in/out
        gained[:window_size] *= fade[:window_size]
        gained[-window_size:] *= fade[window_size:]
        
        return gained

    def check_and_adjust_volume(self, waveform):
        """
        Additional check to catch any remaining volume issues
        """
        # Calculate RMS energy in small windows
        window_size = int(0.100 * self.target_sr)  # 100ms windows
        hop_size = window_size // 2
        
        rms_values = []
        for i in range(0, len(waveform) - window_size, hop_size):
            window = waveform[i:i + window_size]
            rms = np.sqrt(np.mean(window ** 2))
            rms_values.append(rms)
        
        # If we detect any outlier segments
        if np.max(rms_values) > np.mean(rms_values) * 3:
            # Apply additional gentle compression
            return self._dynamic_compress(waveform)
        
        return waveform

    def _dynamic_compress(self, waveform):
        """
        Apply gentle dynamic compression where needed
        """
        processor = Pedalboard([
            Compressor(
                threshold_db=-24,
                ratio=1.5,
                attack_ms=40,
                release_ms=400
            )
        ])
        return processor(waveform, self.target_sr)

    def clean_audio(self, waveform, sr):
        """
        Enhanced cleaning pipeline with better audio preservation
        """
        try:
            # First ensure correct sample rate
            if sr != self.target_sr:
                waveform = librosa.resample(waveform, orig_sr=sr, target_sr=self.target_sr)

            # Remove DC offset
            waveform = waveform - np.mean(waveform)

            # Normalize input level before processing
            waveform = librosa.util.normalize(waveform, norm=np.inf)

            # Apply gentle noise reduction with more conservative parameters
            try:
                cleaned = nr.reduce_noise(
                    y=waveform,
                    sr=self.target_sr,
                    stationary=True,
                    prop_decrease=0.5,  # More conservative noise reduction
                    n_std_thresh_stationary=1.5
                )
            except Exception as e:
                print(f"Noise reduction failed, using original. Error: {str(e)}")
                cleaned = waveform

            # Apply processing chain
            processor = self._get_processor()
            enhanced = processor(cleaned, self.target_sr)

            # Apply peak normalization
            peak = np.abs(enhanced).max()
            if peak > 0:
                enhanced = enhanced * (0.95 / peak)  # Normalize to -0.5dB

            # Ensure audio isn't silent
            if np.abs(enhanced).max() < 0.01:
                print("Warning: Audio too quiet, using normalized original")
                return librosa.util.normalize(waveform)

            return enhanced

        except Exception as e:
            print(f"Audio processing failed: {str(e)}")
            # Return safely normalized original audio as fallback
            return librosa.util.normalize(waveform)



    def process_and_verify(self, waveform, sr):
        """
        Process audio with enhanced verification
        """
        try:
            # Basic validity checks
            if len(waveform) == 0 or not np.any(waveform):
                print("Warning: Empty or invalid audio detected")
                return np.zeros(self.min_audio_length)

            # Store original audio properties
            original_peak = np.abs(waveform).max()
            original_rms = np.sqrt(np.mean(waveform**2))

            # Process the audio
            processed = self.clean_audio(waveform, sr)

            # Verify the processed audio isn't degraded
            processed_peak = np.abs(processed).max()
            processed_rms = np.sqrt(np.mean(processed**2))

            # If processing severely degraded the audio, use normalized original
            if processed_rms < original_rms * 0.3 or processed_peak < original_peak * 0.3:
                print("Warning: Processing degraded audio quality, using normalized original")
                return librosa.util.normalize(waveform)

            return processed

        except Exception as e:
            print(f"Audio verification failed: {str(e)}")
            return librosa.util.normalize(waveform)
