from .reservoir_buffer import ReservoirBuffer
from .scale_buffer import Memory
from .minred_buffer import MinRedBuffer
from .fifo_buffer import FIFOBuffer

def get_buffer(buffer_type: str,
               mem_size: int = 2000,
               alpha_ema: int = 0.5,
               device: str = 'cpu',
               ):
    
    if buffer_type == 'reservoir':
        return ReservoirBuffer(mem_size, alpha_ema, device=device)
    elif buffer_type == 'fifo':
        return FIFOBuffer(mem_size, alpha_ema)
    elif buffer_type == 'minred':
        return MinRedBuffer(mem_size, alpha_ema, device=device)
    elif buffer_type == 'scale':
        return Memory(mem_size=mem_size, device=device)
    elif buffer_type == 'climb':
        return None
    else:
        raise Exception(f'Buffer type {buffer_type} is not supported')