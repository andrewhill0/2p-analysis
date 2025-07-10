import numpy as np

def merge_gloms(F_array_in, roi_mask, num_merged_rois):
    F_array_out = np.zeros((len(F_array_in), num_merged_rois))

    # L1 + R1 (ROI 8 + 9)   ->  F_array_out[ ,0]  ->  + π/8
    # L8 + R2 (ROI 1 + 10)  ->  F_array_out[ ,1]  ->  +3π/8
    # L7 + R3 (    2 + 11)  ->  F_array_out[ ,2]  ->  +5π/8
    # L6 + R4 (    3 + 12)  ->  F_array_out[ ,3]  ->  +7π/8
    # L5 + R5 (    4 + 13)                    4   ->  -7π/8
    # L4 + R6 (    5 + 14)                    5   ->  -5π/8
    # L3 + R7 (    6 + 15)                    6   ->  -3π/8
    # L2 + R8 (    7 + 16)                    7   ->  - π/8

    for r in range(num_merged_rois):
        mask_r = r + 1
        if r == num_merged_rois - 1:
            num_px_L_bridge = np.count_nonzero(roi_mask == mask_r)
            num_px_R_bridge = np.count_nonzero(roi_mask == mask_r + 1)
            L_weight = num_px_L_bridge / (num_px_L_bridge + num_px_R_bridge)
            R_weight = num_px_R_bridge / (num_px_L_bridge + num_px_R_bridge)
            F_array_out[:, 0] = (F_array_in[:, r] * L_weight) + (F_array_in[:, r + 1] * R_weight)
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
    glom_rads = [np.pi*1/8,
                np.pi*3/8,
                np.pi*5/8,
                np.pi*7/8,
                -np.pi*7/8,
                -np.pi*5/8,
                -np.pi*3/8,
                -np.pi*1/8]
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