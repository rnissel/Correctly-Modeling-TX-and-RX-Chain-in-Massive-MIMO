# =============================================================================
# ============================  Ronald Nissel =================================
# ======= Huawei Technologies, Gothenburg Research Center, Sweden  ===========
# ======================== First version: 14.03.2022 ==========================
# =============================================================================
# This script allows to reproduce the numerical results of R. Nissel, "Correctly Modeling TX and RX Chain in (Distributed) Massive MIMO - New Fundamental Insights on Coherency", IEEE Communications Letters, 2022
# I limited myself here to a very simple simulation setup, to make the paper and code as easy to follow as possible
# To make the simulations more realistic, one needs:
# 1) a more realistic channel model, e.g., Quadriga + Polarization + Antenna pattern + three sectors per site;
# 2) Full time-frequency grid e.g., 3300 subcarriers + 1400 OFDM symbols;
# 3) Perform "realistic" channel estimation and calibration (i.e., SRS and DMRS are at certain time-frequency position and noise and interference will cause errors);
# 4) Include switching time between UL and DL transmission;
# 5) Include coding and decoding + (different MCS) and not only rate calculations

import numpy as np
import matplotlib.pyplot as plt
import time

# ==============================================================
# ========================  Variables   ========================
# ==============================================================
CorrectTXandRXmodel = True                         # Either use the correct TRX model (TX:+phi, RX:-phi) or the wrong one (TX:+phi, RX+phi)
CalibrateUEphase = True                            # Performs full reciprocity calibration, i.e., also the UE reciprocity factor is taken into account, which requires in practice DMRS + feedback of the UE channel estimate to the BS

# Timing Variables
NrTimePoints = 101                                  # Number of sample points in time. Note that one sample point can be seen as several OFDM symbols ... but this detail is not simulated
SamplingTime = 0.1e-3                               # Sampling time, i.e., time between two sampling points
Index_Calibration = np.array((20, 80))              # Sampling points at which calibration is performed. Note that the calibration overhead is ignored.
Index_SRS = np.array((0, 20, 40, 60, 80))           # Sampling points at which SRS are transmitted, i.e., perfect UL channel is available. Note that any estimation overhead and error is ignored.

# Phase Noise Parameters [Wiener Process]
# Reference paper: Sigma2 = 10**(-4) at a sampling time of Ts = 10**(-6), see H. Mehrpouyan et al., “Joint estimation of channel and oscillator phase noise in MIMO systems,” IEEE Transactions on Signal Processing, vol. 60, no. 9, pp. 4790–4807, 2012.
Sigma2_PhaseNoiseIncrement = 10**(-4)/10**(-6) * SamplingTime

# Setup
Setup = 'DMM'                   # 'DMM' (joint transmission of 4TRPs, ignore interference to neighboring clusters), 'FullDMIMO' (joint transmission of all TRPs in the network), 'CMM' (seperate precoding at each TRP)
ISD = 200                       # Inter side distance [m]
NrTRXperTRP = 64                # Number of TRX per TRP
NrUEsPerTRP = 10                # Number of UEs per TRP (have highest channel gain to this TRP)
NrRepetitions_LargeScale = 5    # Number of Monte Carlo Repetition for large scale parameters, i.e., path loss
NrRepetitions_SmallScale = 10   # Number of Monte Carlo Repetition for small scale parameters (Rayleigh fading, phase noise)

# Power Settings
P_noise = 290 * 1.3805e-23 * 30e3 * 10**(5/10)          # Noise power for one subcarrier, 5dB NF
PmaxPerTRP = 10**(20/10)*1e-3/(27*12)                   # Signal Power per TRP (20dBm=0.1W distributed over 27*12=324 subcarriers)

# Define number of TRPs and DMIMO Cluster Size
if Setup == 'DMM':
    NrTRPsPerCluster = 4            # Number of Base stations within DMIMO cluster
    NrDMMClusters = 4               # Number of Clusters in total. Note, each Cluster consist of "NrTRPsPerCluster" TRPs, must be x**2
elif Setup == 'FullDMIMO':
    NrTRPsPerCluster = 16           # Number of Base stations within DMIMO cluster
    NrDMMClusters = 1               # Number of Clusters in total. Note, each Cluster consist of "NrTRPsPerCluster" TRPs, must be x**2
elif Setup == 'CMM':
    NrTRPsPerCluster = 1            # Number of Base stations within DMIMO cluster
    NrDMMClusters = 16              # Number of Clusters in total. Note, each Cluster consist of "NrTRPsPerCluster" TRPs, must be x**2


# ==============================================================
# ==================== Preallocate Result  =====================
# ==============================================================
NrTRPs = NrDMMClusters*NrTRPsPerCluster
NrUEs = NrUEsPerTRP*NrTRPs

Signal_FreeRunning = np.zeros((NrTimePoints, NrUEs, NrRepetitions_LargeScale, NrRepetitions_SmallScale), dtype=np.complex_)
SignalAndInterferencePower_FreeRunning = np.zeros((NrTimePoints, NrUEs, NrRepetitions_LargeScale, NrRepetitions_SmallScale))

Signal_Locked = np.zeros((NrTimePoints, NrUEs, NrRepetitions_LargeScale, NrRepetitions_SmallScale), dtype=np.complex_)
SignalAndInterferencePower_Locked = np.zeros((NrTimePoints, NrUEs, NrRepetitions_LargeScale, NrRepetitions_SmallScale))


# ==============================================================
# ====================== Set up Geometry  ======================
# ==============================================================
TRP_position_oneCluster = []
for i_x in range(int(np.sqrt(NrTRPsPerCluster))):
    for i_y in range(int(np.sqrt(NrTRPsPerCluster))):
        TRP_position_oneCluster.append((ISD/2+i_x*ISD)+1j*(ISD/2+i_y*ISD))
TRP_position_oneCluster = np.array(TRP_position_oneCluster)
TRP_position = []
for i_x in range(int(np.sqrt(NrDMMClusters))):
    for i_y in range(int(np.sqrt(NrDMMClusters))):
        TRP_position.append(TRP_position_oneCluster + ((i_x * ISD) + 1j * (i_y * ISD))*np.sqrt(NrTRPsPerCluster))
TRP_position = np.array(TRP_position).ravel()


# ==============================================================
# ====================== Start Simulation  =====================
# ==============================================================
i_loop = 0
t = time.time()
for i_repLarge in range(NrRepetitions_LargeScale):
    # Generate Random Channel Gain: I assign exactly "NrUEsPerTRP" UEs to each TRP, i.e., highest channel gain
    AllocatedUEs = np.zeros(NrTRPs)
    ChannelGain = np.zeros((NrUEs, NrTRPs))
    UE_position = np.zeros(NrUEs, dtype=np.complex_)
    i_UE = 0
    for i_TRP in range(NrTRPs):
        while AllocatedUEs[i_TRP] < NrUEsPerTRP:
            UE_position_temp = TRP_position[i_TRP] + ((np.random.rand(1)-0.5) + 1j*(np.random.rand(1)-0.5))*ISD*1.1
            # Check distance bigger than 25m
            if np.abs(UE_position_temp-TRP_position[i_TRP]) > 25:
                UE_position_temp_wrapAround = UE_position_temp + np.sqrt(NrTRPs) * ISD * np.array((0+0j, 1+0j, 1+1j, 0+1j, -1+1j, -1+0j, -1-1j, 0-1j, 1-1j))
                distances = np.abs(UE_position_temp_wrapAround[:, np.newaxis] - TRP_position[np.newaxis, :])
                UEbelongsToWrongCell = True
                while UEbelongsToWrongCell:
                    # Channel Model From "Bj¨ornson, M. Matthaiou, A. Pitarokoilis, and E. Larsson, “Distributed massive MIMO in cellular networks: Impact of imperfect hardware and number of oscillators,” in EUSIPCO, 2015, pp. 2436–2440
                    shadowFading = 10**(np.sqrt(3.16)*np.random.randn(distances.shape[0], distances.shape[1])-1.53)
                    ChannelGain_temp = np.sum(shadowFading * distances**(-3.76), axis=0)

                    # # Simplified 3GPP Channel Model:
                    # PL1 = 28+22*np.log10(distances) + 20*np.log10(3.5) + np.random.randn(distances.shape[0], distances.shape[1])*4
                    # PL_UMA_NLOS = 13.54 + 39.08*np.log10(distances) + 20*np.log10(3.5) + np.random.randn(distances.shape[0], distances.shape[1])*6
                    # PL = np.maximum(PL1, PL_UMA_NLOS)
                    # ChannelGain_temp = np.sum(10**(-PL/10), axis=0)

                    if np.argmax(ChannelGain_temp) == i_TRP:
                        ChannelGain[i_UE, :] = ChannelGain_temp
                        UE_position[i_UE] = UE_position_temp[0]
                        AllocatedUEs[i_TRP] = AllocatedUEs[i_TRP] + 1
                        i_UE = i_UE+1
                        UEbelongsToWrongCell = False

    # plt.plot(np.real(TRP_position), np.imag(TRP_position),'o')
    # plt.plot(np.real(UE_position), np.imag(UE_position),'.')
    # plt.show()

    for i_repSmall in range(NrRepetitions_SmallScale):
        # Random physical air channel
        H_PHY = np.kron(np.sqrt(ChannelGain), np.ones((1, NrTRXperTRP))) * (np.sqrt(1 / 2) * (np.random.randn(NrUEs, NrTRPs * NrTRXperTRP) + 1j * np.random.randn(NrUEs, NrTRPs * NrTRXperTRP)))

        # Determines which antenna element belongs to which DMIMO cluster
        Cluster_Antenna_Mapping = np.kron(np.arange(NrDMMClusters), np.ones((1, NrTRXperTRP*NrTRPsPerCluster), dtype=int)).ravel()

        # Determines which UE belongs to which cluster
        UE_Cluster_Mapping = np.kron(np.eye(NrDMMClusters), np.ones((NrUEsPerTRP*NrTRPsPerCluster, 1))) == 1

        # Initial LO Phases
        phi_UE = np.random.rand(NrUEs) * 2 * np.pi
        phi_TRP_FreeRunning = np.random.rand(NrTRPs*NrTRXperTRP) * 2 * np.pi
        phi_TRP_Locked = phi_TRP_FreeRunning.copy()

        # Initial Precoder set to zero
        Precoder_FreeRunning = np.zeros((NrTRPs*NrTRXperTRP, NrUEs), dtype=np.complex_)
        Precoder_Locked = np.zeros((NrTRPs*NrTRXperTRP, NrUEs), dtype=np.complex_)

        # Calibration Factors
        RelativeCalibrationFactor_FreeRunning = np.ones(NrTRPs*NrTRXperTRP, dtype=np.complex_)
        PerfectUECalibrationFactor_FreeRunning = np.ones(NrUEs, dtype=np.complex_)
        RelativeCalibrationFactor_Locked = np.ones(NrTRPs*NrTRXperTRP, dtype=np.complex_)
        PerfectUECalibrationFactor_Locked = np.ones(NrUEs, dtype=np.complex_)

        # Sweep over time
        for i_n in range(NrTimePoints):
            # Transmit Chains
            t_UE = np.exp(1j * phi_UE)
            t_TRP_FreeRunning = np.exp(1j * phi_TRP_FreeRunning)
            t_TRP_Locked = np.exp(1j * phi_TRP_Locked)

            # Receive Chains
            if CorrectTXandRXmodel:
                r_UE = np.exp(-1j * phi_UE)
                r_TRP_FreeRunning = np.exp(-1j * phi_TRP_FreeRunning)
                r_TRP_Locked = np.exp(-1j * phi_TRP_Locked)
            else:
                r_UE = t_UE.copy()
                r_TRP_FreeRunning = t_TRP_FreeRunning.copy()
                r_TRP_Locked = t_TRP_Locked.copy()

            # UL and DL channel
            H_UL_FreeRunning = r_TRP_FreeRunning[:, np.newaxis] * H_PHY.transpose() * t_UE[np.newaxis, :]
            H_DL_FreeRunning = r_UE[:, np.newaxis] * H_PHY * t_TRP_FreeRunning[np.newaxis, :]

            H_UL_Locked = r_TRP_Locked[:, np.newaxis] * H_PHY.transpose() * t_UE[np.newaxis, :]
            H_DL_Locked = r_UE[:, np.newaxis] * H_PHY * t_TRP_Locked[np.newaxis, :]

            # Update Reciprocity Calibration (at calibration position "Index_Calibration")
            if np.any(i_n == Index_Calibration):
                for i_cluster in range(NrDMMClusters):
                    # Relative Calibration: 1) TRX0 => all other TRX; 2) all other TRX => TRX0. Channel in-between is assumed to be 1 here, as it would anyway cancel out.
                    y1 = r_TRP_FreeRunning[Cluster_Antenna_Mapping == i_cluster] * t_TRP_FreeRunning[Cluster_Antenna_Mapping == i_cluster][0]
                    y2 = r_TRP_FreeRunning[Cluster_Antenna_Mapping == i_cluster][0] * t_TRP_FreeRunning[Cluster_Antenna_Mapping == i_cluster]
                    RelativeCalibrationFactor_FreeRunning[Cluster_Antenna_Mapping == i_cluster] = y2/y1

                    # UE level perfect reciprocity calibration. Note that to get this factor, the UE needs to feedback information => high overhead
                    z1 = t_TRP_FreeRunning[Cluster_Antenna_Mapping == i_cluster][0] * r_UE[UE_Cluster_Mapping[:, i_cluster]]
                    z2 = r_TRP_FreeRunning[Cluster_Antenna_Mapping == i_cluster][0] * t_UE[UE_Cluster_Mapping[:, i_cluster]]
                    PerfectUECalibrationFactor_FreeRunning[UE_Cluster_Mapping[:, i_cluster]] = z1/z2

                    # Relative Calibration: 1) TRX0 => all other TRX; 2) all other TRX => TRX0. Channel in-between is assumed to be 1 here, as it would anyway cancel out.
                    y1 = r_TRP_Locked[Cluster_Antenna_Mapping == i_cluster] * t_TRP_Locked[Cluster_Antenna_Mapping == i_cluster][0]
                    y2 = r_TRP_Locked[Cluster_Antenna_Mapping == i_cluster][0] * t_TRP_Locked[Cluster_Antenna_Mapping == i_cluster]
                    RelativeCalibrationFactor_Locked[Cluster_Antenna_Mapping == i_cluster] = y2 / y1

                    # UE level perfect reciprocity calibration. Note that to get this factor, the UE needs to feedback information => high overhead
                    z1 = t_TRP_Locked[Cluster_Antenna_Mapping == i_cluster][0] * r_UE[UE_Cluster_Mapping[:, i_cluster]]
                    z2 = r_TRP_Locked[Cluster_Antenna_Mapping == i_cluster][0] * t_UE[UE_Cluster_Mapping[:, i_cluster]]
                    PerfectUECalibrationFactor_Locked[UE_Cluster_Mapping[:, i_cluster]] = z1 / z2

            # Update Precoding Matrix once we get new UL CSI (at SRS position "Index_SRS")
            if np.any(i_n == Index_SRS):
                # Estimate the DL channel based on (perfect) UL channel information + calibration factors
                if CalibrateUEphase:
                    H_DL_estimation_FreeRunning = PerfectUECalibrationFactor_FreeRunning[:, np.newaxis] * H_UL_FreeRunning.transpose() * RelativeCalibrationFactor_FreeRunning[np.newaxis, :]
                    H_DL_estimation_Locked = PerfectUECalibrationFactor_Locked[:, np.newaxis] * H_UL_Locked.transpose() * RelativeCalibrationFactor_Locked[np.newaxis, :]
                else:
                    H_DL_estimation_FreeRunning = H_UL_FreeRunning.transpose() * RelativeCalibrationFactor_FreeRunning[np.newaxis, :]
                    H_DL_estimation_Locked = H_UL_Locked.transpose() * RelativeCalibrationFactor_Locked[np.newaxis, :]

                # Calculate Precoder for each Cluster separately
                for i_cluster in range(NrDMMClusters):
                    H_temp_free = H_DL_estimation_FreeRunning[np.ix_(UE_Cluster_Mapping[:, i_cluster], Cluster_Antenna_Mapping == i_cluster)]
                    Precoder_temp_free = np.matmul(H_temp_free.conj().transpose(), np.linalg.inv(np.matmul(H_temp_free, H_temp_free.conj().transpose())))

                    H_temp_locked = H_DL_estimation_Locked[np.ix_(UE_Cluster_Mapping[:, i_cluster], Cluster_Antenna_Mapping == i_cluster)]
                    Precoder_temp_locked = np.matmul(H_temp_locked.conj().transpose(), np.linalg.inv(np.matmul(H_temp_locked, H_temp_locked.conj().transpose())))

                    Precoder_FreeRunning[np.ix_(Cluster_Antenna_Mapping == i_cluster, UE_Cluster_Mapping[:, i_cluster])] = Precoder_temp_free
                    Precoder_Locked[np.ix_(Cluster_Antenna_Mapping == i_cluster, UE_Cluster_Mapping[:, i_cluster])] = Precoder_temp_locked

                # Normalize precoder and account for TX power
                Precoder_FreeRunning = Precoder_FreeRunning / np.sqrt(np.sum(np.abs(Precoder_FreeRunning) ** 2, axis=0, keepdims=True)) * np.sqrt(PmaxPerTRP / (NrUEsPerTRP))
                Precoder_Locked = Precoder_Locked / np.sqrt(np.sum(np.abs(Precoder_Locked) ** 2, axis=0, keepdims=True)) * np.sqrt(PmaxPerTRP / (NrUEsPerTRP))

            # Calculate Signal and Signal+Interference Power
            HtimesW_FreeRunning = np.matmul(H_DL_FreeRunning, Precoder_FreeRunning)
            Signal_FreeRunning[i_n, :, i_repLarge, i_repSmall] = np.diag(HtimesW_FreeRunning)
            SignalAndInterferencePower_FreeRunning[i_n, :, i_repLarge, i_repSmall] = np.sum(np.abs(HtimesW_FreeRunning) ** 2, axis=1)

            HtimesW_Locked = np.matmul(H_DL_Locked, Precoder_Locked)
            Signal_Locked[i_n, :, i_repLarge, i_repSmall] = np.diag(HtimesW_Locked)
            SignalAndInterferencePower_Locked[i_n, :, i_repLarge, i_repSmall] = np.sum(np.abs(HtimesW_Locked) ** 2, axis=1)

            # Phase drift process
            phi_UE = phi_UE + np.random.randn(NrUEs) * np.sqrt(Sigma2_PhaseNoiseIncrement)
            phi_TRP_FreeRunning = phi_TRP_FreeRunning + np.random.randn(NrTRPs * NrTRXperTRP) * np.sqrt(Sigma2_PhaseNoiseIncrement)
            phi_TRP_Locked = phi_TRP_Locked + np.sqrt(Sigma2_PhaseNoiseIncrement) * np.kron(np.random.randn(NrDMMClusters), np.ones((1, NrTRXperTRP * NrTRPsPerCluster))).ravel()


        # Check time
        elapsed = time.time() - t
        i_loop = i_loop + 1
        TimeLeft = elapsed / i_loop * (NrRepetitions_SmallScale * NrRepetitions_LargeScale - i_loop)
        print(round(i_loop / (NrRepetitions_SmallScale * NrRepetitions_LargeScale) * 100, 1), '%. Time left: ', round(TimeLeft / 60, 1), 'minutes')


# ==============================================================
# ================== Calculate Mean UE Rate  ===================
# ==============================================================

# Assuming DMRS (i.e., perfect effective DL channel at UE side)
SINR_FreeRunning_DMRS = np.abs(Signal_FreeRunning)**2/(SignalAndInterferencePower_FreeRunning - np.abs(Signal_FreeRunning)**2 + P_noise)
SINR_Locked_DMRS = np.abs(Signal_Locked)**2/(SignalAndInterferencePower_Locked - np.abs(Signal_Locked)**2 + P_noise)

MeanUErate_FreeRunning_DMRS = np.mean(np.mean(np.mean(np.minimum(np.log2(1 + SINR_FreeRunning_DMRS), 8), axis=3), axis=2), axis=1)
MeanUErate_Locked_DMRS = np.mean(np.mean(np.mean(np.minimum(np.log2(1 + SINR_Locked_DMRS), 8), axis=3), axis=2), axis=1)

# Blind DL channel estimation = only statistical information at the UE available, see other papers, e.g., [6] or [7]. I personally do not like this rate definition.
SINR_FreeRunning_Blind = np.abs(np.mean(Signal_FreeRunning, axis=3))**2/(np.mean(SignalAndInterferencePower_FreeRunning, axis=3) - np.abs(np.mean(Signal_FreeRunning, axis=3))**2 + P_noise)
SINR_Locked_Blind = np.abs(np.mean(Signal_Locked, axis=3))**2/(np.mean(SignalAndInterferencePower_Locked, axis=3) - np.abs(np.mean(Signal_Locked, axis=3))**2 + P_noise)

MeanUErate_FreeRunning_Blind = np.mean(np.mean(np.minimum(np.log2(1 + SINR_FreeRunning_Blind), 8), axis=2), axis=1)
MeanUErate_Locked_Blind = np.mean(np.mean(np.minimum(np.log2(1 + SINR_Locked_Blind), 8), axis=2), axis=1)


# ==============================================================
# ====================== Plot Figure  ==========================
# ==============================================================
time_ms = np.arange(NrTimePoints)*SamplingTime/1e-3

plt.figure(figsize=(6, 3))
plt.plot(time_ms, MeanUErate_FreeRunning_DMRS, 'r')
plt.plot(time_ms, MeanUErate_Locked_DMRS, 'b')
plt.plot(time_ms, MeanUErate_FreeRunning_Blind, 'r--')
plt.plot(time_ms, MeanUErate_Locked_Blind, 'b--')
plt.ylim((0, 6))
plt.xlim((time_ms[0], time_ms[-1]))
plt.xlabel('Time [ms]')
plt.ylabel('Mean UE Rate [bps/Hz]')

# plt.grid()
if CorrectTXandRXmodel:
    plt.title('Correct TX and RX Model!')
else:
    plt.title('Wrong TX and RX Model!')
plt.tight_layout()
plt.show()

