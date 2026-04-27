import streamlit as st
import librosa
import numpy as np
import pyloudnorm as pyln
import matplotlib.pyplot as plt
import io

# Page Configuration
st.set_page_config(page_title="Audio Analyzer", layout="wide")

# Custom CSS for better UI
# st.markdown("""
#     <style>
#     .main { background-color: #0e1117; }
#     .stMetric { background-color: #161b22; padding: 15px; border-radius: 10px; border: 1px solid #30363d; }
#     </style>
#     """, unsafe_allow_html=True)

st.markdown("""
    <style>
    :root {
        --bg: #0a0a09;
        --bg2: #111110;
        --amber: #e8a020;
        --text: #e8e4d8;
        --border: #2a2a26;
    }
    .stApp { background-color: var(--bg); color: var(--text); font-family: 'DM Sans', sans-serif; }
    h1 { color: #e8a020 !important; font-family: 'Bebas Neue', sans-serif !important; }
    h2, h3 { font-family: 'Bebas Neue', sans-serif !important; color: var(--text); }
    div[data-testid="stFileUploader"] label { 
        color: #e8e4d8 !important; 
        font-family: 'DM Sans', sans-serif !important;
    }    
    [data-testid="stMetric"] { background-color: var(--bg2) !important; border: 1px solid var(--border) !important; }
    [data-testid="stMetricLabel"] { color: var(--text) !important; font-family: 'DM Sans', sans-serif; font-size: 0.9rem; }
    [data-testid="stMetricValue"] { color: var(--amber) !important; font-family: 'DM Mono', monospace; }
    /* Uploader e Bottoni */
    div[data-testid="stFileUploader"] label { color: var(--text) !important; }
    [data-testid="stFileUploader"] button p { color: var(--amber) !important; }
    </style>
    """, unsafe_allow_html=True)

st.title("VENTENA STUDIOS TOOLKIT", width="stretch", text_alignment="center")
st.title("🎵 Audio Analyzer Plus 🎵", width="stretch", text_alignment="center")
st.write("Professional mix and mastering diagnostic tool. Upload your track to get an instant technical report.")

uploaded_file = st.file_uploader("Upload your track (WAV or MP3)", type=["wav", "mp3"])

if uploaded_file is not None:
    st.audio(uploaded_file)
    with st.spinner('Analyzing audio signal... please wait.'):
        # 1. Load Audio (Stereo is mandatory for correlation)
        audio_bytes = uploaded_file.read()
        data, rate = librosa.load(io.BytesIO(audio_bytes), sr=None, mono=False)
        
        # Ensure data is in (Samples, Channels) format for analysis
        if data.ndim == 1:
            st.warning("⚠️ The uploaded file is Mono. Stereo Correlation analysis cannot be performed.")
            is_stereo = False
            data_stereo = data.reshape(-1, 1)
        else:
            is_stereo = True
            data_stereo = data.T # Transpose to (Samples, 2)

        # 2. Loudness Analysis (LUFS & LRA)
        meter = pyln.Meter(rate)
        integrated_loudness = meter.integrated_loudness(data_stereo)
        lra = meter.loudness_range(data_stereo)

        # 3. Peak and Crest Factor
        peak_linear = np.max(np.abs(data))

        # CORREZIONE: Pyloudnorm non ha una funzione diretta "get_true_peak", 
        # quindi aggiungiamo una funzione di stima corretta:
        def get_true_peak(data, rate):
            # Oversampling di 4x per catturare gli inter-sample peaks
            from scipy.signal import resample_poly
            data_oversampled = resample_poly(data, 4, 1)
            return 20 * np.log10(np.max(np.abs(data_oversampled)))

        true_peak_db = get_true_peak(data_stereo, rate)
        
        # OLD true_peak_db = 20 * np.log10(peak_linear) if peak_linear > 0 else -100
        
        rms_level = np.sqrt(np.mean(data**2))
        crest_factor = peak_linear / rms_level if rms_level > 0 else 0

        # 4. Phase Correlation (only for stereo)
        correlation = 0
        if is_stereo:
            # Calculate Pearson correlation coefficient between L and R
            correlation = np.corrcoef(data[0], data[1])[0, 1]

        # --- UI DISPLAY WITH COMPARATIVE REFERENCE TABLES ---
        st.subheader("📊 Quantitative Analysis")
        
        m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
        
        # Helper function to generate reference table
        def get_ref_table(data):
            return "<table style='width:100%; font-size:11px;'><tr><th>Genre</th><th>Range</th></tr>" + \
                   "".join([f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in data.items()]) + "</table>"

        # 1. Loudness
        m_col1.metric("Loudness", f"{integrated_loudness:.1f} LUFS")
        m_col1.markdown(get_ref_table({
            "Pop/EDM": "-10 to -8",
            "Hip-Hop": "-11 to -9",
            "Rock": "-12 to -10",
            "Jazz/Class.": "-18 to -14"
        }), unsafe_allow_html=True)
        
        # 2. True Peak
        m_col2.metric("True Peak", f"{true_peak_db:.2f} dBTP")
        m_col2.caption("Global Std: <= -1.0")
        
        # 3. LU Range (LRA)
        m_col3.metric("LU Range", f"{lra:.1f} LU")
        m_col3.markdown(get_ref_table({
            "Jazz/Class.": "8 to 15",
            "Rock": "5 to 8",
            "Hip-Hop": "4 to 7",
            "Pop/EDM": "3 to 6"
        }), unsafe_allow_html=True)
        
        # 4. Crest Factor
        m_col4.metric("Crest Factor", f"{crest_factor:.2f}")
        m_col4.markdown(get_ref_table({
            "Jazz/Class.": "8 to 12",
            "Rock": "6 to 8",
            "Hip-Hop": "5 to 7",
            "Pop/EDM": "4 to 6"
        }), unsafe_allow_html=True)
        
        # 5. Correlation
        m_col5.metric("Correlation", f"{correlation:.2f}")
        m_col5.caption("Ideal Range: 0.2 to 0.9")

        st.divider()

        # --- TECHNICAL FEEDBACK ---
        st.subheader("💡 Technical Feedback & Improvements")
        
        col_fb1, col_fb2 = st.columns(2)

        with col_fb1:
            st.markdown("### 🎚️ Dynamic & Level")
            # Loudness Logic
            if integrated_loudness > -10:
                st.error("**Oversquashed:** Your track is very loud. It will be turned down by streaming platforms. Try to ease up on the Limiter.")
            elif integrated_loudness < -15:
                st.warning("**Low Level:** The track might sound quiet. You have headroom to increase the loudness for a more competitive master.")
            else:
                st.success("**Target Reached:** Your loudness is ideal for modern streaming standards.")

            # Peak Logic
            if true_peak_db > -0.5:
                st.error("**Clipping Risk:** Peaks are too close to 0dB. Lower your Limiter's Ceiling to -1.0 dBTP to avoid inter-sample peaks.")
            
            # LRA Logic
            if lra < 4:
                st.info("**Small LRA:** Very consistent volume, typical of heavy EDM/Pop. If this is Jazz/Rock, you might be over-compressing.")

        with col_fb2:
            st.markdown("### 🧬 Stereo & Phase")
            if is_stereo:
                if correlation < 0:
                    st.error("**Phase Issues:** Negative correlation detected. Your track will lose significant elements (like bass or vocals) when played in Mono.")
                elif correlation < 0.4:
                    st.warning("**Wide/Thin:** Low correlation. The mix is very wide, but check for mono compatibility.")
                else:
                    st.success("**Solid Phase:** Good correlation. The track will sound great even on mono speakers (phones, clubs).")
            else:
                st.info("Mono file: No phase correlation available.")

        # --- VISUALIZATION ---
        st.subheader("📈 Waveform Visualization")
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.set_facecolor('#0e1117')
        fig.patch.set_facecolor('#0e1117')
        
        if is_stereo:
            librosa.display.waveshow(data[0], sr=rate, ax=ax, alpha=0.5, label='Left', color='#00d1ff')
            librosa.display.waveshow(data[1], sr=rate, ax=ax, alpha=0.5, label='Right', color='#ff007c')
        else:
            librosa.display.waveshow(data, sr=rate, ax=ax, color='#00d1ff')
        
        ax.legend()
        ax.tick_params(colors='white')
        st.pyplot(fig)

      # --- ENHANCED: SPECTRUM ANALYSIS (EQ) WITH REFERENCE ---
        st.subheader("🔊 Spectral Frequency Analysis")
        st.write("Average frequency balance vs. Pink Noise Reference (-3dB/octave).")
        
        # Calculate FFT
        n_fft = 2048
        spec = np.abs(librosa.stft(data[0] if is_stereo else data, n_fft=n_fft))
        mean_spec = np.mean(spec, axis=1)
        freqs = librosa.fft_frequencies(sr=rate, n_fft=n_fft)
        
        # Prepare data for plotting
        db_spec = 20 * np.log10(mean_spec[1:] + 1e-6)
        
        # Create Reference Curve (Pink Noise slope: -3dB per octave)
        # We normalize the reference to match the peak of the track
        ref_curve = -3 * np.log2(freqs[1:] / 1000) + (np.max(db_spec) - 10)
        
        # Plot
        fig_spec, ax_spec = plt.subplots(figsize=(12, 4))
        ax_spec.set_facecolor('#0e1117')
        fig_spec.patch.set_facecolor('#0e1117')
        
        # Plot track spectrum
        ax_spec.semilogx(freqs[1:], db_spec, color='#00ff9f', label='Track Spectrum', linewidth=1.5)
        # Plot reference curve
        ax_spec.semilogx(freqs[1:], ref_curve, color='#ff007c', linestyle='--', label='Target Balance (-3dB/oct)', alpha=0.7)
        
        # Highlights at 20Hz and 20kHz
        for f in [20, 20000]:
            ax_spec.axvline(x=f, color='white', linestyle=':', alpha=0.5)
            ax_spec.text(f, ax_spec.get_ylim()[0], f' {f}Hz', color='white', fontsize=8, rotation=90)
            
        ax_spec.set_xlabel('Frequency (Hz)', color='white')
        ax_spec.set_ylabel('Magnitude (dB)', color='white')
        ax_spec.set_xlim([20, 20000])
        ax_spec.tick_params(colors='white')
        ax_spec.grid(True, which='both', linestyle='--', alpha=0.2)
        ax_spec.legend(loc='upper right')
        
        st.pyplot(fig_spec)
        
        st.info("Analysis: If your green line stays close to the pink dashed line, your mix balance is likely correct. "
                "Large deviations indicate masking, lack of body, or excessive harshness.")
st.caption("Audio Analyzer Plus | Developed by VentenaStudios for Producers & Engineers | 100% Private Analysis")
