import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftshift
from scipy.signal.windows import hann, hamming, blackman

# --- PART 1: SIGNAL GENERATION AND DFT SPECTRUM ANALYSIS (Task 1) ---
f1, f2, f3 = 500.0, 500.25, 499.75
xmax = 3.0
fs = 800.0
N = 1800

A1 = xmax
A2 = 0.1 * A1
A3 = 0.01 * A1

k = np.arange(N)
x = A1 * np.sin(2 * np.pi * f1 / fs * k) + \
    A2 * np.sin(2 * np.pi * f2 / fs * k) + \
    A3 * np.sin(2 * np.pi * f3 / fs * k)

w_rect = np.ones(N)
w_hann = hann(N, sym=False)
w_hamm = hamming(N, sym=False)
w_black = blackman(N, sym=False)

def fft2db(X_raw):
    N_size = X_raw.size
    Xtmp = (2.0 / N_size) * np.abs(X_raw)
    Xtmp *= 0.5
    if N_size % 2 == 0:
        Xtmp[N_size // 2] /= 2.0
    return 20 * np.log10(Xtmp + 1e-10)

df = fs / N
freqs = np.arange(N) * df

X_rect = fft(x * w_rect)
X_hann = fft(x * w_hann)
X_hamm = fft(x * w_hamm)
X_black = fft(x * w_black)

windows_data = [
    ('Rectangular Window', X_rect, 'C0'),
    ('Hann Window', X_hann, 'C1'),
    ('Hamming Window', X_hamm, 'C2'),
    ('Blackman Window', X_black, 'C3')
]

plt.figure(figsize=(12, 10))
for i, (name, X_w, color) in enumerate(windows_data, 1):
    plt.subplot(4, 1, i)
    plt.plot(freqs, fft2db(X_w), color=color, label=name)
    plt.xlim(290, 310)  # Zoom around the aliased frequency at 300 Hz
    plt.ylim(-60, 15)
    plt.title(f'DFT Spectrum - {name}')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Amplitude [dB]')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig('task1_signal_spectrum.png', dpi=300)
plt.show()

# --- PART 2: WINDOW FUNCTION DTFT SPECTRA (Task 2) ---
def windDTFTdB(w):
    N_len = w.size
    Nz = 100 * N_len
    W = np.zeros(Nz)
    W[0:N_len] = w
    W_fft = np.abs(fftshift(fft(W)))
    W_fft /= np.max(W_fft)
    W_db = 20 * np.log10(W_fft + 1e-10)
    Omega = 2 * np.pi / Nz * np.arange(Nz) - np.pi
    return Omega, W_db

plt.figure(figsize=(10, 6))
for w, label, color in [(w_rect, 'rect', 'C0'), (w_hann, 'hann', 'C1'), (w_hamm, 'hamming', 'C2'), (w_black, 'blackman', 'C3')]:
    Omega, W_db = windDTFTdB(w)
    plt.plot(Omega, W_db, label=label, color=color)

plt.xlim(-0.03, 0.03)  # Zoom into the main lobe
plt.ylim(-120, 10)
plt.xlabel(r'$\Omega$ [rad]')
plt.ylabel(r'$|W(\Omega)|$ [dB]')
plt.title('Window Functions DTFT Spectra Normalized to Main Lobe Maximum')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('task2_window_characteristics.png', dpi=300)
plt.show()