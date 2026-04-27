import streamlit as st
import librosa
import numpy as np
import pyloudnorm as pyln
import matplotlib.pyplot as plt
import io
from scipy.signal import resample_poly

# Page Configuration
st.set_page_config(page_title="Audio Analyzer — AudioToolkit", layout="wide")

# CSS Custom per conformità a INDEX.txt
st.markdown("""
    <style>
    :root {
        --bg: #0a0a09;
        --bg2: #111110;
        --amber: #e8a020;
        --text: #e8e4d8;
        --muted: #7a7870;
        --border: #2a2a26;
    }
    .main { background-color: var(--bg); color: var(--text); font-family: 'DM Sans', sans-serif; }
    
    /* Titoli */
    h1, h2, h3 { font-family: 'Bebas Neue', sans-serif !important; color: var(--text); letter-spacing: 0.05em; }
    h1 { color: var(--amber) !important; font-size: 3rem !important; }
    
    /* Metric Cards */
    [data-testid="stMetricValue"] { color: var(--amber) !important; font-family: 'DM Mono', monospace; }
    [data-testid="stMetricLabel"] { color: var(--muted) !important; text-transform: uppercase; font-size: 0.7rem; letter-spacing: 0.1em; }
    
    /* Box & Divs */
    div.stButton > button { border: 1px solid var(--amber); background: transparent; color: var(--amber); }
    .stApp { background-color: var(--bg); }
    div[data-testid="stMetric"] { background-color: var(--bg2); border: 1px solid var(--border); padding: 20px; border-radius: 0; }
    
    /* Plot styling */
    .stPlotlyChart { border: 1px solid var(--border); }
    </style>
    """, unsafe_allow_html=True)

st.title("AUDIO TOOLKIT")
st.subheader("🎵 Mix & Master Analyzer")
st.markdown("<p style='color: var(--muted);'>Professional tools for your mix. Real-time diagnostic.</p>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload your track (WAV or MP3)", type=["wav", "mp3"])

if uploaded_file is not None:
    with st.spinner('Analyzing audio signal...'):
        audio_bytes = uploaded_file.read()
        data, rate = librosa.load(io.BytesIO(audio_bytes), sr=None, mono=False)
        
        if data.ndim == 1:
            is_stereo = False
            data_stereo = data.reshape(-1, 1)
        else:
            is_stereo = True
            data_stereo = data.T

        # Analysis
        meter = pyln.Meter(rate)
        integrated_loudness = meter.integrated_loudness(data_stereo)
        lra = meter.loudness_range(data_stereo)

        def get_true_peak(data, rate):
            data_oversampled = resample_poly(data, 4, 1)
            return 20 * np.log10(np.max(np.abs(data_oversampled)))

        true_peak_db = get_true_peak(data_stereo, rate)
        rms_level = np.sqrt(np.mean(data**2))
        crest_factor = (np.max(np.abs(data)) / rms_level) if rms_level > 0 else 0
        correlation = np.corrcoef(data[0], data[1])[0, 1] if is_stereo else 0

        # UI Display
        st.markdown("### 📊 Quantitative Analysis")
        m1, m2, m3, m4, m5 = st.columns(5)
        
        m1.metric("Loudness", f"{integrated_loudness:.1f} LUFS")
        m2.metric("True Peak", f"{true_peak_db:.2f} dBTP")
        m3.metric("LU Range", f"{lra:.1f} LU")
        m4.metric("Crest Factor", f"{crest_factor:.2f}")
        m5.metric("Correlation", f"{correlation:.2f}")

        st.divider()

        # Feedback & Visuals
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 💡 Technical Feedback")
            if integrated_loudness > -10: st.error("Oversquashed: Level too high for streaming platforms.")
            elif integrated_loudness < -15: st.warning("Low Level: You have headroom to increase gain.")
            else: st.success("Target LUFS reached.")
        
        with col2:
            st.markdown("### 🧬 Stereo & Phase")
            if is_stereo:
                if correlation < 0: st.error("Phase issues detected!")
                else: st.success("Solid phase.")

        st.subheader("📈 Spectral Analysis")
        fig, ax = plt.subplots(figsize=(10, 3))
        fig.patch.set_facecolor('#0a0a09')
        ax.set_facecolor('#111110')
        
        spec = np.abs(librosa.stft(data[0] if is_stereo else data, n_fft=2048))
        db_spec = 20 * np.log10(np.mean(spec, axis=1) + 1e-6)
        freqs = librosa.fft_frequencies(sr=rate, n_fft=2048)
        
        ax.semilogx(freqs[1:], db_spec[1:], color='#e8a020')
        ax.tick_params(colors='#e8e4d8')
        ax.grid(True, alpha=0.1)
        st.pyplot(fig)

st.caption("AudioToolkit Analyzer — Developed by Ventena Studios")
