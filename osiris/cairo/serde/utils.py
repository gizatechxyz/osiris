def to_fp(value, fp_impl='FP16x16'):
    sign = 0 if value >= 0 else 1

    match fp_impl:
        case 'FP16x16':
            return (abs(int(value * 2**16)), sign)
        case 'FP8x23':
            return (abs(int(value * 2**23)), sign)
        case 'FP32x32':
            return (abs(int(value * 2**32)), sign)
        case 'FP64x64':
            return (abs(int(value * 2**64)), sign)


def from_fp(value, fp_impl='FP16x16'):
    match fp_impl:
        case 'FP16x16':
            return value / 2**16
        case 'FP8x23':
            return value / 2**23
        case 'FP32x32':
            return value / 2**32
        case 'FP64x64':
            return value / 2**64
