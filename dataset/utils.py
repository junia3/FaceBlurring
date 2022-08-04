import numpy as np
import math

def psnr(source, target):
    '''
        Calculate PSNR between source and target image
    '''
    mse = np.mean((source - target) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    PSNR = 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

    return PSNR


