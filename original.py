import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import requests
import tempfile
import os
from scipy import signal
from scipy.signal import savgol_filter
from matplotlib.widgets import SpanSelector, Button
import matplotlib.patches as patches

class InteractiveAudioAnalyzer:
    def __init__(self, audio_path, sr=22050):
        self.audio_path = audio_path
        self.y, self.sr = librosa.load(audio_path, sr=sr)
        self.duration = len(self.y) / self.sr
        self.zoom_start = 0
        self.zoom_end = self.duration
        
        print(f"โหลดไฟล์: {audio_path}")
        print(f"Sample rate: {self.sr} Hz, ความยาว: {self.duration:.2f} วินาที")
    
    def plot_waveform_analysis(self):
        """
        แสดงกราฟ waveform แบบต่างๆ พร้อมฟังก์ชันซูม
        """
        fig, axes = plt.subplots(4, 2, figsize=(16, 12))
        fig.suptitle('การวิเคราะห์ Waveform และ Envelope', fontsize=16, fontweight='bold')
        
        # เวลาทั้งหมด
        t_full = np.linspace(0, self.duration, len(self.y))
        
        # 1. Raw Waveform
        axes[0,0].plot(t_full, self.y, color='blue', linewidth=0.5)
        axes[0,0].set_title('1. Raw Waveform (คลื่นเสียงดิบ)')
        axes[0,0].set_ylabel('Amplitude')
        axes[0,0].grid(True, alpha=0.3)
        axes[0,0].set_xlim(0, self.duration)
        
        # 2. Amplitude Envelope
        envelope = np.abs(signal.hilbert(self.y))
        axes[0,1].plot(t_full, envelope, color='red', linewidth=1)
        axes[0,1].fill_between(t_full, envelope, alpha=0.3, color='red')
        axes[0,1].set_title('2. Amplitude Envelope (ซองขีดคลื่น)')
        axes[0,1].set_ylabel('Amplitude')
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. RMS Energy
        frame_length = 2048
        hop_length = 512
        rms = librosa.feature.rms(y=self.y, frame_length=frame_length, hop_length=hop_length)[0]
        t_rms = librosa.frames_to_time(np.arange(len(rms)), sr=self.sr, hop_length=hop_length)
        
        axes[1,0].plot(t_rms, rms, color='green', linewidth=2)
        axes[1,0].fill_between(t_rms, rms, alpha=0.4, color='green')
        axes[1,0].set_title('3. RMS Energy (พลังงานเฉลี่ย)')
        axes[1,0].set_ylabel('RMS')
        axes[1,0].grid(True, alpha=0.3)
        
        # 4. Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(self.y, frame_length=frame_length, hop_length=hop_length)[0]
        t_zcr = librosa.frames_to_time(np.arange(len(zcr)), sr=self.sr, hop_length=hop_length)
        
        axes[1,1].plot(t_zcr, zcr, color='purple', linewidth=2)
        axes[1,1].set_title('4. Zero Crossing Rate (อัตราผ่านศูนย์)')
        axes[1,1].set_ylabel('ZCR')
        axes[1,1].grid(True, alpha=0.3)
        
        # 5. Spectral Centroid
        spectral_centroids = librosa.feature.spectral_centroid(y=self.y, sr=self.sr, hop_length=hop_length)[0]
        t_sc = librosa.frames_to_time(np.arange(len(spectral_centroids)), sr=self.sr, hop_length=hop_length)
        
        axes[2,0].plot(t_sc, spectral_centroids, color='orange', linewidth=2)
        axes[2,0].set_title('5. Spectral Centroid (จุดศูนย์กลางความถี่)')
        axes[2,0].set_ylabel('Frequency (Hz)')
        axes[2,0].grid(True, alpha=0.3)
        
        # 6. Pitch (F0)
        pitches, magnitudes = librosa.piptrack(y=self.y, sr=self.sr, hop_length=hop_length)
        pitch_values = []
        pitch_times = []
        
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 0:
                pitch_values.append(pitch)
                pitch_times.append(librosa.frames_to_time(t, sr=self.sr, hop_length=hop_length))
        
        if pitch_values:
            axes[2,1].plot(pitch_times, pitch_values, 'o-', color='red', markersize=2, linewidth=1)
            axes[2,1].set_title('6. Fundamental Frequency (ความถี่พื้นฐาน)')
            axes[2,1].set_ylabel('F0 (Hz)')
            axes[2,1].grid(True, alpha=0.3)
        
        # 7. MFCC (แสดงแค่ 3 coefficients แรก)
        mfccs = librosa.feature.mfcc(y=self.y, sr=self.sr, n_mfcc=3, hop_length=hop_length)
        t_mfcc = librosa.frames_to_time(np.arange(mfccs.shape[1]), sr=self.sr, hop_length=hop_length)
        
        colors_mfcc = ['red', 'blue', 'green']
        for i in range(3):
            axes[3,0].plot(t_mfcc, mfccs[i], color=colors_mfcc[i], linewidth=2, label=f'MFCC {i+1}')
        axes[3,0].set_title('7. MFCC Features (3 coefficients แรก)')
        axes[3,0].set_ylabel('MFCC Value')
        axes[3,0].set_xlabel('เวลา (วินาที)')
        axes[3,0].legend()
        axes[3,0].grid(True, alpha=0.3)
        
        # 8. Onset Detection
        onset_frames = librosa.onset.onset_detect(y=self.y, sr=self.sr, hop_length=hop_length)
        onset_times = librosa.frames_to_time(onset_frames, sr=self.sr, hop_length=hop_length)
        
        # แสดง waveform พร้อม onset markers
        axes[3,1].plot(t_full, self.y, color='blue', alpha=0.6, linewidth=0.5)
        for onset_time in onset_times:
            axes[3,1].axvline(x=onset_time, color='red', linestyle='--', alpha=0.8)
        axes[3,1].set_title('8. Onset Detection (จุดเริ่มเสียง)')
        axes[3,1].set_ylabel('Amplitude')
        axes[3,1].set_xlabel('เวลา (วินาที)')
        axes[3,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # เพิ่มฟังก์ชันซูม
        self.setup_zoom_selector(axes[0,0], t_full, self.y)
        
        plt.show()
        
        return fig, axes
    
    def setup_zoom_selector(self, ax, time_data, audio_data):
        """
        ตั้งค่า zoom selector สำหรับกราฟ
        """
        def onselect(xmin, xmax):
            self.zoom_start = max(0, xmin)
            self.zoom_end = min(self.duration, xmax)
            print(f"เลือกช่วงเวลา: {self.zoom_start:.2f} - {self.zoom_end:.2f} วินาที")
            self.plot_zoomed_analysis()
        
        # สร้าง SpanSelector
        span = SpanSelector(ax, onselect, 'horizontal', useblit=True,
                          props=dict(alpha=0.3, facecolor='yellow'))
        
        # เพิ่มข้อความแนะนำ
        ax.text(0.02, 0.95, 'ลากเมาส์เพื่อเลือกช่วงที่ต้องการซูม', 
                transform=ax.transAxes, fontsize=10, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    def plot_zoomed_analysis(self):
        """
        แสดงการวิเคราะห์ในช่วงที่ซูม
        """
        # หาดัชนีที่ตรงกับช่วงเวลาที่เลือก
        start_idx = int(self.zoom_start * self.sr)
        end_idx = int(self.zoom_end * self.sr)
        
        y_zoom = self.y[start_idx:end_idx]
        t_zoom = np.linspace(self.zoom_start, self.zoom_end, len(y_zoom))
        
        if len(y_zoom) == 0:
            print("ช่วงเวลาที่เลือกไม่ถูกต้อง")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'การวิเคราะห์แบบซูม: {self.zoom_start:.2f} - {self.zoom_end:.2f} วินาที', 
                    fontsize=14, fontweight='bold')
        
        # 1. Zoomed Waveform
        axes[0,0].plot(t_zoom, y_zoom, color='blue', linewidth=1)
        axes[0,0].set_title('Waveform (ซูม)')
        axes[0,0].set_ylabel('Amplitude')
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. FFT Spectrum (ณ จุดกึ่งกลางของช่วงที่เลือก)
        if len(y_zoom) > 1024:
            mid_idx = len(y_zoom) // 2
            window_size = min(1024, len(y_zoom))
            start_window = max(0, mid_idx - window_size // 2)
            end_window = start_window + window_size
            
            y_window = y_zoom[start_window:end_window]
            fft = np.fft.fft(y_window)
            freqs = np.fft.fftfreq(len(y_window), 1/self.sr)
            
            # แสดงเฉพาะความถี่บวก
            positive_freqs = freqs[:len(freqs)//2]
            magnitude = np.abs(fft[:len(fft)//2])
            
            axes[0,1].plot(positive_freqs, 20*np.log10(magnitude + 1e-10), color='red')
            axes[0,1].set_title('FFT Spectrum (ณ จุดกึ่งกลาง)')
            axes[0,1].set_xlabel('ความถี่ (Hz)')
            axes[0,1].set_ylabel('Magnitude (dB)')
            axes[0,1].grid(True, alpha=0.3)
            axes[0,1].set_xlim(0, self.sr//2)
        
        # 3. Instantaneous Frequency
        analytic_signal = signal.hilbert(y_zoom)
        instantaneous_phase = np.unwrap(np.angle(analytic_signal))
        instantaneous_frequency = (np.diff(instantaneous_phase) / (2.0*np.pi) * self.sr)
        
        if len(instantaneous_frequency) > 0:
            t_if = t_zoom[1:]  # ลดขนาดเพราะ diff
            # กรองค่าที่ผิดปกติ
            valid_mask = (instantaneous_frequency > 0) & (instantaneous_frequency < self.sr//2)
            
            axes[0,2].plot(t_if[valid_mask], instantaneous_frequency[valid_mask], 
                          color='green', linewidth=1)
            axes[0,2].set_title('Instantaneous Frequency')
            axes[0,2].set_ylabel('ความถี่ (Hz)')
            axes[0,2].grid(True, alpha=0.3)
        
        # 4. Short-time Energy
        window_length = min(256, len(y_zoom)//10)
        if window_length > 0:
            energy = []
            energy_times = []
            
            for i in range(0, len(y_zoom) - window_length, window_length//4):
                window = y_zoom[i:i+window_length]
                energy.append(np.sum(window**2))
                energy_times.append(t_zoom[i + window_length//2])
            
            if energy:
                axes[1,0].plot(energy_times, energy, color='purple', linewidth=2)
                axes[1,0].fill_between(energy_times, energy, alpha=0.3, color='purple')
                axes[1,0].set_title('Short-time Energy')
                axes[1,0].set_ylabel('Energy')
                axes[1,0].grid(True, alpha=0.3)
        
        # 5. Autocorrelation
        autocorr = np.correlate(y_zoom, y_zoom, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr = autocorr / autocorr[0]  # normalize
        
        lags = np.arange(len(autocorr)) / self.sr * 1000  # แปลงเป็น ms
        
        # แสดงเฉพาะช่วง 0-50ms
        max_lag_ms = min(50, len(autocorr) / self.sr * 1000)
        max_lag_samples = int(max_lag_ms * self.sr / 1000)
        
        axes[1,1].plot(lags[:max_lag_samples], autocorr[:max_lag_samples], color='orange')
        axes[1,1].set_title('Autocorrelation (0-50ms)')
        axes[1,1].set_xlabel('Lag (ms)')
        axes[1,1].set_ylabel('Correlation')
        axes[1,1].grid(True, alpha=0.3)
        
        # 6. Spectrogram ของช่วงที่ซูม
        if len(y_zoom) > 256:
            f, t_spec, Sxx = signal.spectrogram(y_zoom, self.sr, nperseg=256, noverlap=128)
            t_spec = t_spec + self.zoom_start  # ปรับเวลาให้ตรงกับช่วงจริง
            
            im = axes[1,2].pcolormesh(t_spec, f, 10*np.log10(Sxx + 1e-10), cmap='viridis')
            axes[1,2].set_title('Spectrogram (ซูม)')
            axes[1,2].set_ylabel('ความถี่ (Hz)')
            axes[1,2].set_xlabel('เวลา (วินาที)')
            plt.colorbar(im, ax=axes[1,2], label='Power (dB)')
        
        plt.tight_layout()
        plt.show()
    
    def plot_pitch_contour_analysis(self):
        """
        แสดงการวิเคราะห์ Pitch Contour แบบละเอียด
        """
        fig, axes = plt.subplots(3, 2, figsize=(16, 12))
        fig.suptitle('การวิเคราะห์ Pitch Contour', fontsize=16, fontweight='bold')
        
        hop_length = 512
        
        # 1. Pitch tracking ด้วยวิธี piptrack
        print("กำลังคำนวณ pitch ด้วย piptrack...")
        pitches, magnitudes = librosa.piptrack(y=self.y, sr=self.sr, hop_length=hop_length, threshold=0.1)
        
        # แยก pitch values ที่มีความเชื่อมั่นสูง
        pitch_values = []
        pitch_times = []
        pitch_confidences = []
        
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            confidence = magnitudes[index, t]
            
            if pitch > 0 and confidence > 0.1:  # กรองเฉพาะที่มีความเชื่อมั่น
                pitch_values.append(pitch)
                pitch_times.append(librosa.frames_to_time(t, sr=self.sr, hop_length=hop_length))
                pitch_confidences.append(confidence)
        
        # แปลงเป็น numpy arrays
        pitch_times = np.array(pitch_times)
        pitch_values = np.array(pitch_values)
        pitch_confidences = np.array(pitch_confidences)
        
        # กรองข้อมูล outliers
        if len(pitch_values) > 0:
            # กำจัด outliers ด้วย IQR method
            Q1 = np.percentile(pitch_values, 25)
            Q3 = np.percentile(pitch_values, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            valid_mask = (pitch_values >= lower_bound) & (pitch_values <= upper_bound) & (pitch_values >= 50) & (pitch_values <= 1000)
            
            pitch_times_clean = pitch_times[valid_mask]
            pitch_values_clean = pitch_values[valid_mask]
            pitch_confidences_clean = pitch_confidences[valid_mask]
        else:
            pitch_times_clean = np.array([])
            pitch_values_clean = np.array([])
            pitch_confidences_clean = np.array([])
        
        # 1. Raw Pitch Contour
        if len(pitch_values_clean) > 0:
            # สี based on confidence
            scatter = axes[0,0].scatter(pitch_times_clean, pitch_values_clean, 
                                      c=pitch_confidences_clean, cmap='viridis', 
                                      s=20, alpha=0.7)
            plt.colorbar(scatter, ax=axes[0,0], label='Confidence')
            
            # เส้นเชื่อม
            axes[0,0].plot(pitch_times_clean, pitch_values_clean, 'b-', alpha=0.5, linewidth=1)
            
        axes[0,0].set_title('1. Raw Pitch Contour (with Confidence)')
        axes[0,0].set_ylabel('Frequency (Hz)')
        axes[0,0].grid(True, alpha=0.3)
        axes[0,0].set_ylim(50, 1000)
        
        # 2. Smoothed Pitch Contour
        if len(pitch_values_clean) > 5:
            from scipy.signal import savgol_filter
            # Smooth pitch contour
            window_length = min(11, len(pitch_values_clean) if len(pitch_values_clean) % 2 == 1 else len(pitch_values_clean) - 1)
            if window_length >= 3:
                pitch_smoothed = savgol_filter(pitch_values_clean, window_length, 3)
                axes[0,1].plot(pitch_times_clean, pitch_values_clean, 'lightblue', alpha=0.5, label='Raw')
                axes[0,1].plot(pitch_times_clean, pitch_smoothed, 'red', linewidth=2, label='Smoothed')
                axes[0,1].legend()
        
        axes[0,1].set_title('2. Smoothed Pitch Contour')
        axes[0,1].set_ylabel('Frequency (Hz)')
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. Pitch in Musical Notes
        if len(pitch_values_clean) > 0:
            # แปลงเป็นโน้ตดนตรี
            note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            
            def hz_to_note(freq):
                if freq <= 0:
                    return ""
                A4 = 440
                C0 = A4*np.power(2, -4.75)
                
                if freq > C0:
                    h = round(12*np.log2(freq/C0))
                    octave = h // 12
                    n = h % 12
                    return note_names[n] + str(octave)
                return ""
            
            # สร้าง array ของโน้ต
            notes = [hz_to_note(freq) for freq in pitch_values_clean]
            
            # แสดงเฉพาะโน้ตที่ไม่ซ้ำกันติดต่อกัน
            unique_notes = []
            unique_times = []
            unique_freqs = []
            
            prev_note = ""
            for i, note in enumerate(notes):
                if note != prev_note and note != "":
                    unique_notes.append(note)
                    unique_times.append(pitch_times_clean[i])
                    unique_freqs.append(pitch_values_clean[i])
                    prev_note = note
            
            if unique_notes:
                for i, (time, freq, note) in enumerate(zip(unique_times, unique_freqs, unique_notes)):
                    axes[1,0].scatter(time, freq, s=100, alpha=0.7)
                    axes[1,0].annotate(note, (time, freq), xytext=(5, 5), 
                                     textcoords='offset points', fontsize=8)
                
                axes[1,0].plot(pitch_times_clean, pitch_values_clean, 'gray', alpha=0.3)
        
        axes[1,0].set_title('3. Pitch as Musical Notes')
        axes[1,0].set_ylabel('Frequency (Hz)')
        axes[1,0].grid(True, alpha=0.3)
        
        # 4. Pitch Statistics and Histogram
        if len(pitch_values_clean) > 0:
            # คำนวณสถิติ
            mean_pitch = np.mean(pitch_values_clean)
            std_pitch = np.std(pitch_values_clean)
            median_pitch = np.median(pitch_values_clean)
            min_pitch = np.min(pitch_values_clean)
            max_pitch = np.max(pitch_values_clean)
            
            # Histogram
            axes[1,1].hist(pitch_values_clean, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            axes[1,1].axvline(mean_pitch, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_pitch:.1f} Hz')
            axes[1,1].axvline(median_pitch, color='green', linestyle='--', linewidth=2, label=f'Median: {median_pitch:.1f} Hz')
            axes[1,1].legend()
            
            # แสดงสถิติ
            stats_text = f"""Pitch Statistics:
Mean: {mean_pitch:.1f} Hz
Median: {median_pitch:.1f} Hz
Std: {std_pitch:.1f} Hz
Min: {min_pitch:.1f} Hz
Max: {max_pitch:.1f} Hz
Range: {max_pitch-min_pitch:.1f} Hz"""
            
            axes[1,1].text(0.02, 0.98, stats_text, transform=axes[1,1].transAxes, 
                          fontsize=9, verticalalignment='top',
                          bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))
        
        axes[1,1].set_title('4. Pitch Distribution')
        axes[1,1].set_xlabel('Frequency (Hz)')
        axes[1,1].set_ylabel('Count')
        
        # 5. Pitch Derivative (ความเร็วการเปลี่ยนแปลง pitch)
        if len(pitch_values_clean) > 1:
            time_diff = np.diff(pitch_times_clean)
            pitch_diff = np.diff(pitch_values_clean)
            
            # คำนวณอัตราการเปลี่ยนแปลง (Hz/s)
            valid_time_mask = time_diff > 0
            if np.any(valid_time_mask):
                pitch_rate = pitch_diff[valid_time_mask] / time_diff[valid_time_mask]
                pitch_rate_times = pitch_times_clean[1:][valid_time_mask]
                
                axes[2,0].plot(pitch_rate_times, pitch_rate, 'purple', linewidth=1.5)
                axes[2,0].axhline(0, color='black', linestyle='-', alpha=0.3)
                axes[2,0].fill_between(pitch_rate_times, pitch_rate, 0, alpha=0.3, color='purple')
        
        axes[2,0].set_title('5. Pitch Rate of Change (Hz/s)')
        axes[2,0].set_ylabel('Rate (Hz/s)')
        axes[2,0].set_xlabel('เวลา (วินาที)')
        axes[2,0].grid(True, alpha=0.3)
        
        # 6. Vibrato Analysis (การสั่นของ pitch)
        if len(pitch_values_clean) > 10:
            # หา vibrato โดยดูการสั่นของ pitch
            if len(pitch_values_clean) > 5:
                window_length = min(11, len(pitch_values_clean) if len(pitch_values_clean) % 2 == 1 else len(pitch_values_clean) - 1)
                if window_length >= 3:
                    pitch_smoothed = savgol_filter(pitch_values_clean, window_length, 3)
                    vibrato = pitch_values_clean - pitch_smoothed
                    
                    axes[2,1].plot(pitch_times_clean, vibrato, 'orange', linewidth=1.5)
                    axes[2,1].axhline(0, color='black', linestyle='-', alpha=0.3)
                    axes[2,1].fill_between(pitch_times_clean, vibrato, 0, alpha=0.3, color='orange')
                    
                    # คำนวณค่า RMS ของ vibrato
                    vibrato_rms = np.sqrt(np.mean(vibrato**2))
                    axes[2,1].text(0.02, 0.98, f'Vibrato RMS: {vibrato_rms:.2f} Hz', 
                                  transform=axes[2,1].transAxes, fontsize=10,
                                  bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))
        
        axes[2,1].set_title('6. Vibrato Analysis (Pitch Deviation)')
        axes[2,1].set_ylabel('Deviation (Hz)')
        axes[2,1].set_xlabel('เวลา (วินาที)')
        axes[2,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return pitch_times_clean, pitch_values_clean
        """
        แสดงการวิเคราะห์ในโดเมนความถี่
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('การวิเคราะห์ในโดเมนความถี่', fontsize=16, fontweight='bold')
        
        # คำนวณ FFT
        fft = np.fft.fft(self.y)
        freqs = np.fft.fftfreq(len(self.y), 1/self.sr)
        
        # แสดงเฉพาะความถี่บวก
        positive_freqs = freqs[:len(freqs)//2]
        magnitude = np.abs(fft[:len(fft)//2])
        phase = np.angle(fft[:len(fft)//2])
        
        # 1. Magnitude Spectrum (Linear)
        axes[0,0].plot(positive_freqs, magnitude, color='blue', linewidth=0.5)
        axes[0,0].set_title('1. Magnitude Spectrum (Linear)')
        axes[0,0].set_xlabel('ความถี่ (Hz)')
        axes[0,0].set_ylabel('Magnitude')
        axes[0,0].grid(True, alpha=0.3)
        axes[0,0].set_xlim(0, self.sr//2)
        
        # 2. Magnitude Spectrum (dB)
        magnitude_db = 20 * np.log10(magnitude + 1e-10)
        axes[0,1].plot(positive_freqs, magnitude_db, color='red', linewidth=0.5)
        axes[0,1].set_title('2. Magnitude Spectrum (dB)')
        axes[0,1].set_xlabel('ความถี่ (Hz)')
        axes[0,1].set_ylabel('Magnitude (dB)')
        axes[0,1].grid(True, alpha=0.3)
        axes[0,1].set_xlim(0, self.sr//2)
        
        # 3. Phase Spectrum
        axes[0,2].plot(positive_freqs, phase, color='green', linewidth=0.5)
        axes[0,2].set_title('3. Phase Spectrum')
        axes[0,2].set_xlabel('ความถี่ (Hz)')
        axes[0,2].set_ylabel('Phase (radians)')
        axes[0,2].grid(True, alpha=0.3)
        axes[0,2].set_xlim(0, self.sr//2)
        
        # 4. Power Spectral Density
        f_psd, psd = signal.welch(self.y, self.sr, nperseg=1024)
        axes[1,0].semilogy(f_psd, psd, color='purple')
        axes[1,0].set_title('4. Power Spectral Density')
        axes[1,0].set_xlabel('ความถี่ (Hz)')
        axes[1,0].set_ylabel('PSD (V²/Hz)')
        axes[1,0].grid(True, alpha=0.3)
        
        # 5. Cepstrum
        log_spectrum = np.log(magnitude + 1e-10)
        cepstrum = np.fft.ifft(log_spectrum).real
        quefrency = np.arange(len(cepstrum)) / self.sr * 1000  # ms
        
        # แสดงเฉพาะช่วง 0-50ms
        max_quefrency_ms = 50
        max_samples = int(max_quefrency_ms * self.sr / 1000)
        
        axes[1,1].plot(quefrency[:max_samples], cepstrum[:max_samples], color='orange')
        axes[1,1].set_title('5. Cepstrum (0-50ms)')
        axes[1,1].set_xlabel('Quefrency (ms)')
        axes[1,1].set_ylabel('Amplitude')
        axes[1,1].grid(True, alpha=0.3)
        
        # 6. Spectral Features Summary
        # คำนวณ spectral features
        spectral_centroid = np.sum(positive_freqs * magnitude) / np.sum(magnitude)
        spectral_bandwidth = np.sqrt(np.sum(((positive_freqs - spectral_centroid) ** 2) * magnitude) / np.sum(magnitude))
        spectral_rolloff_idx = np.where(np.cumsum(magnitude) >= 0.85 * np.sum(magnitude))[0]
        spectral_rolloff = positive_freqs[spectral_rolloff_idx[0]] if len(spectral_rolloff_idx) > 0 else 0
        
        # แสดงค่าต่างๆ
        features_text = f"""Spectral Features:
        
Centroid: {spectral_centroid:.1f} Hz
Bandwidth: {spectral_bandwidth:.1f} Hz
Rolloff (85%): {spectral_rolloff:.1f} Hz
Peak Frequency: {positive_freqs[np.argmax(magnitude)]:.1f} Hz
Total Energy: {np.sum(magnitude**2):.2e}"""
        
        axes[1,2].text(0.1, 0.7, features_text, transform=axes[1,2].transAxes, 
                      fontsize=10, verticalalignment='top',
                      bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        axes[1,2].set_title('6. Spectral Features Summary')
        axes[1,2].axis('off')
        
        plt.tight_layout()
        plt.show()

def main():
    print("=== Interactive Audio Analyzer ===")
    print("เลือกโหมดการวิเคราะห์:")
    print("1. ไฟล์เดียว (ไฟล์ในเครื่อง)")
    print("2. ไฟล์เดียว (URL)")
    print("3. เปรียบเทียบเสียงคน vs AI (ไฟล์ในเครื่อง)")
    
    mode_choice = input("เลือก (1-3): ").strip()
    
    if mode_choice == "1":
        audio_file = input("ใส่ path ของไฟล์เสียง: ").strip()
        audio_path = audio_file
        analyzer = InteractiveAudioAnalyzer(audio_path)
        run_single_analysis(analyzer)
        
    elif mode_choice == "2":
        audio_url = input("ใส่ URL ของไฟล์เสียง: ").strip()
        audio_path, temp_file = download_audio(audio_url)
        if audio_path:
            analyzer = InteractiveAudioAnalyzer(audio_path)
            run_single_analysis(analyzer)
            if temp_file:
                os.unlink(temp_file)
        
    elif mode_choice == "3":
        print("\n=== เปรียบเทียบเสียงคน vs AI (ไฟล์ในเครื่อง) ===")
        human_file = input("ใส่ path ของไฟล์เสียงคนจริง: ").strip()
        ai_file = input("ใส่ path ของไฟล์เสียง AI: ").strip()
        
        try:
            run_comparison_analysis(human_file, ai_file)
        except FileNotFoundError as e:
            print(f"ไม่พบไฟล์: {e}")
        except Exception as e:
            print(f"เกิดข้อผิดพลาด: {e}")
    else:
        print("กรุณาเลือก 1-3")

def download_audio(url):
    """ดาวน์โหลดไฟล์เสียงจาก URL"""
    try:
        print(f"กำลังดาวน์โหลดไฟล์จาก: {url}")
        response = requests.get(url)
        response.raise_for_status()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(response.content)
            return tmp_file.name, tmp_file.name
        
    except Exception as e:
        print(f"ไม่สามารถดาวน์โหลดได้: {e}")
        return None, None

def run_single_analysis(analyzer):
    """รันการวิเคราะห์ไฟล์เดียว"""
    while True:
        print("\n=== เลือกการวิเคราะห์ ===")
        print("1. Waveform Analysis (พร้อมฟังก์ชันซูม)")
        print("2. Pitch Contour Analysis (วิเคราะห์ความถี่พื้นฐาน)")
        print("3. Frequency Domain Analysis")
        print("4. ออกจากโปรแกรม")
        
        analysis_choice = input("เลือก (1-4): ").strip()
        
        if analysis_choice == "1":
            print("กำลังสร้าง Waveform Analysis...")
            print("หลังจากกราฟแสดง ให้ลากเมาส์บนกราฟแรกเพื่อเลือกช่วงที่ต้องการซูม")
            analyzer.plot_waveform_analysis()
            
        elif analysis_choice == "2":
            print("กำลังสร้าง Pitch Contour Analysis...")
            print("กำลังวิเคราะห์ pitch... (อาจใช้เวลาสักครู่)")
            analyzer.plot_pitch_contour_analysis()
            
        elif analysis_choice == "3":
            print("กำลังสร้าง Frequency Domain Analysis...")
            analyzer.plot_frequency_domain_analysis()
            
        elif analysis_choice == "4":
            break
        else:
            print("กรุณาเลือก 1-4")

def run_comparison_analysis(human_path, ai_path):
    """รันการวิเคราะห์เปรียบเทียบ"""
    print("กำลังโหลดไฟล์เสียงทั้งคู่...")
    human_analyzer = InteractiveAudioAnalyzer(human_path)
    ai_analyzer = InteractiveAudioAnalyzer(ai_path)
    
    while True:
        print("\n=== เลือกการเปรียบเทียบ ===")
        print("1. เปรียบเทียบ Waveform")
        print("2. เปรียบเทียบ Pitch Contour")
        print("3. เปรียบเทียบ Frequency Domain")
        print("4. ออกจากโปรแกรม")
        
        comparison_choice = input("เลือก (1-4): ").strip()
        
        if comparison_choice == "1":
            print("กำลังเปรียบเทียบ Waveform...")
            compare_waveforms(human_analyzer, ai_analyzer)
            
        elif comparison_choice == "2":
            print("กำลังเปรียบเทียบ Pitch Contour...")
            print("กำลังวิเคราะห์ pitch... (อาจใช้เวลาสักครู่)")
            compare_pitch_contours(human_analyzer, ai_analyzer)
            
        elif comparison_choice == "3":
            print("กำลังเปรียบเทียบ Frequency Domain...")
            compare_frequency_domains(human_analyzer, ai_analyzer)
            
        elif comparison_choice == "4":
            break
        else:
            print("กรุณาเลือก 1-4")

def compare_waveforms(human_analyzer, ai_analyzer):
    """เปรียบเทียบ waveform ระหว่างเสียงคนกับ AI"""
    fig, axes = plt.subplots(4, 2, figsize=(16, 12))
    fig.suptitle('เปรียบเทียบ Waveform Analysis: เสียงคน vs AI', fontsize=16, fontweight='bold')
    
    analyzers = [human_analyzer, ai_analyzer]
    labels = ['เสียงคนจริง (Human)', 'เสียง AI (AI)']
    colors = [['blue', 'red', 'green', 'purple'], ['darkblue', 'darkred', 'darkgreen', 'indigo']]
    
    for idx, (analyzer, label, color_set) in enumerate(zip(analyzers, labels, colors)):
        y, sr = analyzer.y, analyzer.sr
        duration = analyzer.duration
        t_full = np.linspace(0, duration, len(y))
        
        hop_length = 512
        
        # 1. Raw Waveform
        axes[0, idx].plot(t_full, y, color=color_set[0], linewidth=0.5)
        axes[0, idx].set_title(f'1. Raw Waveform ({label})')
        axes[0, idx].set_ylabel('Amplitude')
        axes[0, idx].grid(True, alpha=0.3)
        axes[0, idx].set_xlim(0, duration)
        
        # 2. RMS Energy
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop_length)[0]
        t_rms = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
        
        axes[1, idx].plot(t_rms, rms, color=color_set[1], linewidth=2)
        axes[1, idx].fill_between(t_rms, rms, alpha=0.4, color=color_set[1])
        axes[1, idx].set_title(f'2. RMS Energy ({label})')
        axes[1, idx].set_ylabel('RMS')
        axes[1, idx].grid(True, alpha=0.3)
        
        # 3. Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(y, frame_length=2048, hop_length=hop_length)[0]
        t_zcr = librosa.frames_to_time(np.arange(len(zcr)), sr=sr, hop_length=hop_length)
        
        axes[2, idx].plot(t_zcr, zcr, color=color_set[2], linewidth=2)
        axes[2, idx].set_title(f'3. Zero Crossing Rate ({label})')
        axes[2, idx].set_ylabel('ZCR')
        axes[2, idx].grid(True, alpha=0.3)
        
        # 4. Spectral Centroid
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
        t_sc = librosa.frames_to_time(np.arange(len(spectral_centroids)), sr=sr, hop_length=hop_length)
        
        axes[3, idx].plot(t_sc, spectral_centroids, color=color_set[3], linewidth=2)
        axes[3, idx].set_title(f'4. Spectral Centroid ({label})')
        axes[3, idx].set_ylabel('Frequency (Hz)')
        axes[3, idx].set_xlabel('เวลา (วินาที)')
        axes[3, idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def compare_pitch_contours(human_analyzer, ai_analyzer):
    """เปรียบเทียบ pitch contour ระหว่างเสียงคนกับ AI"""
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle('เปรียบเทียบ Pitch Contour Analysis: เสียงคน vs AI', fontsize=16, fontweight='bold')
    
    analyzers = [human_analyzer, ai_analyzer]
    labels = ['เสียงคนจริง (Human)', 'เสียง AI (AI)']
    
    for idx, (analyzer, label) in enumerate(zip(analyzers, labels)):
        y, sr = analyzer.y, analyzer.sr
        hop_length = 512
        
        # คำนวณ pitch
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr, hop_length=hop_length, threshold=0.1)
        
        pitch_values = []
        pitch_times = []
        pitch_confidences = []
        
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            confidence = magnitudes[index, t]
            
            if pitch > 0 and confidence > 0.1:
                pitch_values.append(pitch)
                pitch_times.append(librosa.frames_to_time(t, sr=sr, hop_length=hop_length))
                pitch_confidences.append(confidence)
        
        pitch_times = np.array(pitch_times)
        pitch_values = np.array(pitch_values)
        pitch_confidences = np.array(pitch_confidences)
        
        # กรอง outliers
        if len(pitch_values) > 0:
            Q1 = np.percentile(pitch_values, 25)
            Q3 = np.percentile(pitch_values, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            valid_mask = (pitch_values >= lower_bound) & (pitch_values <= upper_bound) & (pitch_values >= 50) & (pitch_values <= 1000)
            
            pitch_times_clean = pitch_times[valid_mask]
            pitch_values_clean = pitch_values[valid_mask]
            pitch_confidences_clean = pitch_confidences[valid_mask]
        else:
            pitch_times_clean = np.array([])
            pitch_values_clean = np.array([])
            pitch_confidences_clean = np.array([])
        
        # 1. Raw Pitch Contour
        if len(pitch_values_clean) > 0:
            scatter = axes[0, idx].scatter(pitch_times_clean, pitch_values_clean, 
                                         c=pitch_confidences_clean, cmap='viridis', 
                                         s=20, alpha=0.7)
            axes[0, idx].plot(pitch_times_clean, pitch_values_clean, 'b-', alpha=0.5, linewidth=1)
            
            if idx == 1:  # แสดง colorbar เฉพาะด้านขวา
                plt.colorbar(scatter, ax=axes[0, idx], label='Confidence')
        
        axes[0, idx].set_title(f'1. Raw Pitch Contour ({label})')
        axes[0, idx].set_ylabel('Frequency (Hz)')
        axes[0, idx].grid(True, alpha=0.3)
        axes[0, idx].set_ylim(50, 1000)
        
        # 2. Pitch Statistics
        if len(pitch_values_clean) > 0:
            axes[1, idx].hist(pitch_values_clean, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            
            mean_pitch = np.mean(pitch_values_clean)
            median_pitch = np.median(pitch_values_clean)
            
            axes[1, idx].axvline(mean_pitch, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_pitch:.1f} Hz')
            axes[1, idx].axvline(median_pitch, color='green', linestyle='--', linewidth=2, label=f'Median: {median_pitch:.1f} Hz')
            axes[1, idx].legend()
            
            # สถิติ
            std_pitch = np.std(pitch_values_clean)
            min_pitch = np.min(pitch_values_clean)
            max_pitch = np.max(pitch_values_clean)
            
            stats_text = f"""Statistics:
Mean: {mean_pitch:.1f} Hz
Median: {median_pitch:.1f} Hz
Std: {std_pitch:.1f} Hz
Range: {max_pitch-min_pitch:.1f} Hz"""
            
            axes[1, idx].text(0.02, 0.98, stats_text, transform=axes[1, idx].transAxes, 
                            fontsize=9, verticalalignment='top',
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))
        
        axes[1, idx].set_title(f'2. Pitch Distribution ({label})')
        axes[1, idx].set_xlabel('Frequency (Hz)')
        axes[1, idx].set_ylabel('Count')
        
        # 3. Vibrato Analysis
        if len(pitch_values_clean) > 10:
            window_length = min(11, len(pitch_values_clean) if len(pitch_values_clean) % 2 == 1 else len(pitch_values_clean) - 1)
            if window_length >= 3:
                pitch_smoothed = savgol_filter(pitch_values_clean, window_length, 3)
                vibrato = pitch_values_clean - pitch_smoothed
                
                axes[2, idx].plot(pitch_times_clean, vibrato, 'orange', linewidth=1.5)
                axes[2, idx].axhline(0, color='black', linestyle='-', alpha=0.3)
                axes[2, idx].fill_between(pitch_times_clean, vibrato, 0, alpha=0.3, color='orange')
                
                vibrato_rms = np.sqrt(np.mean(vibrato**2))
                axes[2, idx].text(0.02, 0.98, f'Vibrato RMS: {vibrato_rms:.2f} Hz', 
                                transform=axes[2, idx].transAxes, fontsize=10,
                                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))
        
        axes[2, idx].set_title(f'3. Vibrato Analysis ({label})')
        axes[2, idx].set_ylabel('Deviation (Hz)')
        axes[2, idx].set_xlabel('เวลา (วินาที)')
        axes[2, idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def compare_frequency_domains(human_analyzer, ai_analyzer):
    """เปรียบเทียบ frequency domain ระหว่างเสียงคนกับ AI"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('เปรียบเทียบ Frequency Domain Analysis: เสียงคน vs AI', fontsize=16, fontweight='bold')
    
    analyzers = [human_analyzer, ai_analyzer]
    labels = ['เสียงคนจริง (Human)', 'เสียง AI (AI)']
    colors = ['blue', 'red']
    
    for idx, (analyzer, label, color) in enumerate(zip(analyzers, labels, colors)):
        y, sr = analyzer.y, analyzer.sr
        
        # คำนวณ FFT
        fft = np.fft.fft(y)
        freqs = np.fft.fftfreq(len(y), 1/sr)
        
        positive_freqs = freqs[:len(freqs)//2]
        magnitude = np.abs(fft[:len(fft)//2])
        magnitude_db = 20 * np.log10(magnitude + 1e-10)
        
        # 1. Magnitude Spectrum
        axes[0, idx].plot(positive_freqs, magnitude_db, color=color, linewidth=0.5)
        axes[0, idx].set_title(f'Magnitude Spectrum ({label})')
        axes[0, idx].set_xlabel('ความถี่ (Hz)')
        axes[0, idx].set_ylabel('Magnitude (dB)')
        axes[0, idx].grid(True, alpha=0.3)
        axes[0, idx].set_xlim(0, sr//2)
        
        # 2. Power Spectral Density
        f_psd, psd = signal.welch(y, sr, nperseg=1024)
        axes[1, idx].semilogy(f_psd, psd, color=color)
        axes[1, idx].set_title(f'Power Spectral Density ({label})')
        axes[1, idx].set_xlabel('ความถี่ (Hz)')
        axes[1, idx].set_ylabel('PSD (V²/Hz)')
        axes[1, idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()