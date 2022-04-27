import numpy as np
import struct
import gzip

def _load_uint8(f):
    idx_dtype, ndim = struct.unpack('BBBB', f.read(4))[2:]
    shape = struct.unpack('>' + 'I' * ndim, f.read(4 * ndim))
    buffer_length = int(np.prod(shape))
    data = np.frombuffer(f.read(buffer_length), dtype=np.uint8).reshape(shape)
    return data

def _save_uint8(data, f):
    data = np.asarray(data, dtype=np.uint8)
    f.write(struct.pack('BBBB', 0, 0, 0x08, data.ndim))
    f.write(struct.pack('>' + 'I' * data.ndim, *data.shape))
    f.write(data.tobytes())

def save_idx(data: np.ndarray, path: str):
    """Writes an array to disk in IDX format.

    Parameters
    ----------
    data : array_like
        Input array of dtype ``uint8`` (will be coerced if different dtype).
    path : str
        Path of the output file. Will compress with `gzip` if path ends in '.gz'.

    References
    ----------
    http://yann.lecun.com/exdb/mnist/
    """
    open_fcn = gzip.open if path.endswith('.gz') else open
    with open_fcn(path, 'wb') as f:
        _save_uint8(data, f)

def load_idx(path: str) -> np.ndarray:
    """Reads an array in IDX format from disk.

    Parameters
    ----------
    path : str
        Path of the input file. Will uncompress with `gzip` if path ends in '.gz'.

    Returns
    -------
    np.ndarray
        Output array of dtype ``uint8``.

    References
    ----------
    http://yann.lecun.com/exdb/mnist/
    """
    open_fcn = gzip.open if path.endswith('.gz') else open
    with open_fcn(path, 'rb') as f:
        return _load_uint8(f)

def format_setup_spec(model_type, model_args, dataset_names):
    try:
        model_spec = str(model_args)
    except KeyError:
        raise ValueError(f"Invalid model type: '{model_type}'. "
                         f"Expected one of {list(_FORMATTERS.keys())}")
    dataset_spec = '+'.join(dataset_names)
    return f"{model_type}-{model_spec}_{dataset_spec}"

def parse_setup_spec(string):
    match = re.match(r"^(.+)-(.+)_(.+)$", string)
    if match is None:
        raise ValueError(f"Invalid setup spec string: '{string}'")
    model_type, model_spec, dataset_spec = match.group(1), match.group(2), match.group(3)
    try:
        model_args = int(model_spec)
    except KeyError:
        raise ValueError(f"Invalid model type: '{model_type}'. "
                         f"Expected one of {list(_PARSERS.keys())}")
    dataset_names = dataset_spec.split('+')
    return model_type, model_args, dataset_names


"""
    normalizes measurments to between 0 and 1 based on the 
    valeus in the csvs 
"""
def scale_measurement(measurement):
    maxs  = np.array([3.15562500e+02, 8.18154942e+01, 1.10646995e+01, 1.06328181e+00, 2.34247117e+01, 2.05090761e+01]) 
    mins = np.array([23.8125    ,  9.99264069,  1.11092356, -0.88446947, 2.99923749,  8.12636895])
    
    ranges = maxs - mins
    return (measurement - mins)/ranges




