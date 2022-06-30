# =============================================================================
# ============================  Ronald Nissel =================================
# ======= Huawei Technologies, Gothenburg Research Center, Sweden  ===========
# ======================== First version: 14.03.2022 ==========================
# =============================================================================
# This example script implements the appendix of the paper: R. Nissel, "Correctly Modeling TX and RX Chain in (Distributed) Massive MIMO - New Fundamental Insights on Coherency", IEEE Communications Letters, 2022
# It compared the simulated total channel effect with the theoretical predictions (which turns out to be exactly the same, as it should be)

import numpy as np
import matplotlib.pyplot as plt

# ==============================================================
# ========================  Variables   ========================
# ==============================================================
F = 30e3            # Subcarrier spacing [Hz]
L = 20              # Number of Subcarriers
K = 3               # Time length in multiple of 1/F, with 3, I assume "CP + pure OFDM + CS", where the length of the CP (cyclic prefix) and CS (cyclic suffix) are roughly the same as for the "pure" OFDM symbol, i.e., 1/F. Note, that tau_t can cause small differences. Also note that I choose a very long CP and CS for simplification to avoid any problems caused by a too short CP and CS.
fc = 3.5e9          # Carrier frequency [Hz]
fs = 10e9           # Sampling rate, i.e., I approximate the continuous time using a very high sampling rate [Hz]
IFFTimplementation = True  # If set to True, use IFFT implementation instead of the direct analytical calculation used in the paper

# ==============================================================
# =================== Dependent Variables   ====================
# ==============================================================
# Align sampling rate and carrier frequency to better fit sampling rate
fs = np.round(fs/F)*F
fc = np.round(fc/F)*F

# FFT size
N_FFT = np.round(fs/F).astype(int)

# Time variables
T = 1/F
dt = 1/fs
t = np.arange(N_FFT*K).reshape((1, -1))*dt - 1/F

# Subcarrier index
l = np.arange(-L/2, L/2).astype(int).reshape((-1, 1))

# ==============================================================
# ===========  Random TX and RX components   =================
# ==============================================================
# Random delay at TX and RX of OFDM modulation/demodulation. Note: we round so that it fits better the IFFT implementatiob
tau_t = np.round(0.05*np.random.rand(1)*N_FFT)*dt
tau_r = np.round(0.05*np.random.rand(1)*N_FFT)*dt

phi_t = np.random.rand(1)*2*np.pi
phi_r = np.random.rand(1)*2*np.pi

# Very Simple Random impulse responses.
h_t_LF = np.random.rand(int(N_FFT*0.05))/(N_FFT*0.05)
h_t_RF = (np.random.rand(int(N_FFT*0.05)) + np.cos(2*np.pi*fc*t.ravel()[:int(N_FFT*0.05)]))/(N_FFT*0.05)
h = (np.random.rand(int(N_FFT*0.05)) + np.cos(2*np.pi*fc*t.ravel()[:int(N_FFT*0.05)]))/(N_FFT*0.05)
h_r_RF = (np.random.rand(int(N_FFT*0.05)) + np.cos(2*np.pi*fc*t.ravel()[:int(N_FFT*0.05)]))/(N_FFT*0.05)
h_r_LF = np.random.rand(int(N_FFT*0.05))/(N_FFT*0.05)

# ==============================================================
# ========= Start Simulation and Compare with Theory  ==========
# ==============================================================

# 4 QAM data symbols
x_l = (np.random.randint(2, size=(L, 1))-0.5)*2 + 1j * (np.random.randint(2, size=(L, 1))-0.5)*2

# ==============================================================
# ============== OFDM Signal in the Time Domain  ===============
# ==============================================================
print('Calculate s_OFDM .... ')
if IFFTimplementation:
    # x_all consists of both, QAM data and zero subcarriers
    x_all = np.zeros(N_FFT, dtype=np.complex_)
    x_all[l] = x_l
    # CP and suffix means we have to copy the ifft output K times, roll to emulate the time delay
    s_OFDM = np.roll(np.tile(np.fft.ifft(x_all), K), int(tau_t/dt)) * (N_FFT)
else:
    s_OFDM = np.sum(np.exp(1j*2*np.pi*l*F*(t-tau_t)) * x_l, axis=0)


# ==============================================================
# ====================  s1: filter at TX  ======================
# ==============================================================
print('Calculate s_1 ... ')
s1_I = np.convolve(np.real(s_OFDM), h_t_LF)[:N_FFT*K]
s1_Q = np.convolve(np.imag(s_OFDM), h_t_LF)[:N_FFT*K]

H_t_LF = np.fft.fft(h_t_LF, N_FFT)
s1_I_Calculation = np.real(np.sum(np.exp(1j*2*np.pi*l*F*(t-tau_t)) * x_l * H_t_LF[l], axis=0))
s1_Q_Calculation = np.imag(np.sum(np.exp(1j*2*np.pi*l*F*(t-tau_t)) * x_l * H_t_LF[l], axis=0))

fig, axs = plt.subplots(2)
fig.suptitle('Error: Convolution (zero padding) vs Calculation (circular) \n Note: we are interested only at the 33µs in the middle => no error')
axs[0].plot(t.ravel()/1e-6, (s1_I_Calculation-s1_I)/np.sqrt(np.mean(np.abs(s1_I)**2)))
axs[1].plot(t.ravel()/1e-6, (s1_Q_Calculation-s1_Q)/np.sqrt(np.mean(np.abs(s1_Q)**2)))
axs[0].set(ylabel='Relative Error: s_1_I')
axs[1].set(ylabel='Relative Error: s_1_Q')
plt.xlabel('Time [µs]')

# ==============================================================
# ==================  s2: After Upconverting  ==================
# ==============================================================
print('Calculate s_2 ... ')
s2 = s1_I * np.cos(2*np.pi*fc*t.ravel() + phi_t) - s1_Q * np.sin(2*np.pi*fc*t.ravel() + phi_t)

s2_Calculation = np.real(np.exp(1j*phi_t) * np.sum(np.exp(1j*2*np.pi*(l*F+fc)*t) * np.exp(-1j*2*np.pi*l*F*tau_t) * x_l * H_t_LF[l], axis=0))

plt.figure()
plt.plot(t.ravel()/1e-6, (s2_Calculation-s2)/np.sqrt(np.mean(np.abs(s2)**2)))
plt.ylabel('Relative Error: s_2')
plt.xlabel('Time [µs]')
plt.title('Error: Convolution (zero padding) vs Calculation (circular) \n Note: we are interested only at the 33µs in the middle => no error')


# ==============================================================
# ==================  s3: Before Downmixing  ===================
# ==============================================================
print('Calculate s_3 ... ')
s3 = np.convolve(np.convolve(np.convolve(s2, h_t_RF), h), h_r_RF)[:N_FFT*K]

H_t_RF = np.fft.fft(h_t_RF, N_FFT)
H = np.fft.fft(h, N_FFT)
H_r_RF = np.fft.fft(h_r_RF, N_FFT)
s3_Calculation = np.real(np.exp(1j*phi_t) * np.sum(np.exp(1j*2*np.pi*(l*F+fc)*t) * np.exp(-1j*2*np.pi*l*F*tau_t) *
                                                   H_t_LF[l] * H_t_RF[l+int(fc/F)] * H[l+int(fc/F)] * H_r_RF[l+int(fc/F)] * x_l, axis=0))

plt.figure()
plt.plot(t.ravel()/1e-6, (s3_Calculation-s3)/np.sqrt(np.mean(np.abs(s3)**2)))
plt.ylabel('Relative Error: s_3')
plt.xlabel('Time [µs]')
plt.title('Error: Convolution (zero padding) vs Calculation (circular) \n Note: we are interested only at the 33µs in the middle => no error')

# ==============================================================
# ================  s4: Downmixing + LP filter  ================
# ==============================================================
print('Calculate s_4 ... ')
s_4 = np.convolve(h_r_LF, s3 * (np.cos(2*np.pi*fc*t.ravel() + phi_r) - 1j * np.sin(2*np.pi*fc*t.ravel() + phi_r)))[:N_FFT*K]

H_r_LF = np.fft.fft(h_r_LF, N_FFT)
s4_Calculation = (1/2 * np.exp(1j*phi_t) * np.exp(-1j*phi_r) * np.sum(np.exp(1j*2*np.pi*l*F*t) * np.exp(-1j*2*np.pi*l*F*tau_t) *
                                                   H_t_LF[l] * H_t_RF[l+int(fc/F)] * H[l+int(fc/F)] * H_r_RF[l+int(fc/F)] * H_r_LF[l] * x_l, axis=0)) + \
                 (1/2 * np.exp(-1j*4*np.pi*fc*t.ravel()) * np.exp(-1j*phi_r) * np.conj(np.exp(1j*phi_t) * np.sum(np.exp(1j*2*np.pi*l*F*t) * np.exp(-1j*2*np.pi*l*F*tau_t) *
                                                   H_t_LF[l] * H_t_RF[l+int(fc/F)] * H[l+int(fc/F)] * H_r_RF[l+int(fc/F)] * H_r_LF[l+2*int(fc/F)] * x_l, axis=0)))

plt.figure()
plt.plot(t.ravel()/1e-6, np.abs(s4_Calculation-s_4)/np.sqrt(np.mean(np.abs(s_4)**2)))
plt.ylabel('Relative Error: s_4')
plt.xlabel('Time [µs]')
plt.title('Error: Convolution (zero padding) vs Calculation (circular) \n Note: we are interested only at the 33µs in the middle => no error')


# ==============================================================
# ===================  y: OFDM Demodulation  ===================
# ==============================================================
if IFFTimplementation:
    # x_all consists of both, QAM data AND zero subcarriers
    y_all = np.fft.fft(s_4[N_FFT+int(tau_r/dt): 2*N_FFT+int(tau_r/dt)])/N_FFT*2
    y_l = y_all[l]
else:
    y_l = 2*F*np.sum(s_4[N_FFT+int(tau_r/dt): 2*N_FFT+int(tau_r/dt)].reshape((1, -1)) * np.exp(-1j*2*np.pi*l*F*(t[0, N_FFT+int(tau_r/dt): 2*N_FFT+int(tau_r/dt)]-tau_r)), axis=1)*dt

t_l_Calculation = np.exp(1j*phi_t) * np.exp(-1j*2*np.pi*l*F*tau_t) * H_t_LF[l] * H_t_RF[l+int(fc/F)]
r_l_Calculation = np.exp(-1j*phi_r) * np.exp(1j*2*np.pi*l*F*tau_r) * H_r_LF[l] * H_r_RF[l+int(fc/F)]

H_total_Simulation = y_l.ravel()/x_l.ravel()
H_total_Calculation = r_l_Calculation.ravel() * H[l+int(fc/F)].ravel() * t_l_Calculation.ravel()

print('====================================== Calculation vs Simulation ==============================================')
row_format = '{:^20}{:^2}{:^20}{:^20}{:^20}{:^2}{:^20}{:^20}'
print(row_format.format('Subcarrier', '|', 'RX Chain', 'Air Channel', 'TX Chain', '|', 'Total Calc.', 'Total Sim.'))

for i_l in range(L):
    print(row_format.format(str(l[i_l]),
                            '|',
                            str(np.round(r_l_Calculation[i_l], 4)),
                            str(np.round(H[i_l+int(fc/F)].ravel(), 4)),
                            str(np.round(t_l_Calculation[i_l], 4)),
                            '|',
                            str(np.round(H_total_Calculation[i_l], 4)),
                            str(np.round(H_total_Simulation[i_l], 4))))
print('Note: Simulations and calculations are the same, showing that the equations are correct')

plt.show()