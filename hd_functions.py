import numpy as np
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d

def merge_gloms(F_array_in, roi_mask, num_merged_rois):
    F_array_out = np.zeros((len(F_array_in), num_merged_rois))

    # Using Hulse et al numbering of PB glomeruli:
    # L8 + R2 = ROI 1 + 10 --> F_array_out[:, 1] -->
    # L7 + R3 = ROI 2 + 11 --> F_array_out[:, 2] -->
    # L6 + R4 = ROI 3 + 12 --> F_array_out[:, 3] -->
    # L5 + R5 = ROI 4 + 13 --> F_array_out[:, 4] -->
    # L4 + R6 = ROI 5 + 14 --> F_array_out[:, 5] -->
    # L3 + R7 = ROI 6 + 15 --> F_array_out[:, 6] -->
    # L2 + R8 = ROI 7 + 16 --> F_array_out[:, 7] -->
    # L1 + R1 = ROI 8 + 9  --> F_array_out[:, 0] -->

    for r in range(num_merged_rois):
        mask_r = r + 1
        # Handle L1/R1 merge:
        if r == num_merged_rois - 1:
            num_px_L_bridge = np.count_nonzero(roi_mask == mask_r)
            num_px_R_bridge = np.count_nonzero(roi_mask == mask_r + 1)
            L_weight = num_px_L_bridge / (num_px_L_bridge + num_px_R_bridge)
            R_weight = num_px_R_bridge / (num_px_L_bridge + num_px_R_bridge)
            F_array_out[:, 0] = (F_array_in[:, r] * L_weight) + (F_array_in[:, r + 1] * R_weight)
        # All other corresponding glomeruli are 9 apart:
        else:
            num_px_L_bridge = np.count_nonzero(roi_mask == mask_r)
            num_px_R_bridge = np.count_nonzero(roi_mask == mask_r + 9)
            L_weight = num_px_L_bridge / (num_px_L_bridge + num_px_R_bridge)
            R_weight = num_px_R_bridge / (num_px_L_bridge + num_px_R_bridge)
            F_array_out[:, r + 1] = (F_array_in[:, r] * L_weight) + (F_array_in[:, r + 9] * R_weight)
    
    return F_array_out


def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    
    return x, y


def PVA_calc(F_array_in):
    num_rois = len(F_array_in[0,:])
    glom_rads = [np.pi*-7/8,
                 np.pi*-5/8,
                 np.pi*-3/8,
                 np.pi*-1/8,
                 np.pi*1/8,
                 np.pi*3/8,
                 np.pi*5/8,
                 np.pi*7/8]
    PVA_rad = np.zeros(len(F_array_in))
    PVA_str = np.zeros(len(F_array_in))

    for t in range(len(F_array_in)):
        
        x_per_roi = np.zeros(num_rois)
        y_per_roi = np.zeros(num_rois)
        
        for r in range(num_rois):
            x_per_roi[r], y_per_roi[r] = pol2cart(F_array_in[t, r], glom_rads[r])
            
        x_sum = sum(x_per_roi)
        y_sum = sum(y_per_roi)
        PVA_rad[t] = np.arctan2(y_sum, x_sum)
        PVA_str[t] = np.sqrt((x_sum**2) + (y_sum**2))

    return PVA_rad, PVA_str


def low_pass_filter(signal_in, cutoff, sample_rate):
    w = cutoff / (sample_rate/2)
    [b, a] = butter(1, w, 'lowpass')
    signal_out = filtfilt(b, a, signal_in, axis = 0, padtype = 'odd', padlen = 3*(max(len(b), len(a))-1))

    return signal_out


def downsample_to_vols(signal_in, num_cycles):
    # Generate the interp1d function:
    f = interp1d(np.arange(len(signal_in)), signal_in, axis = 0, fill_value = 'extrapolate')
    new_range = np.linspace(0, len(signal_in), num_cycles)
    interp_out = f(new_range)

    return interp_out

