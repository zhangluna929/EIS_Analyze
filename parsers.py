import pandas as pd
from pathlib import Path
import re

__all__ = ['load_eis']


def _load_csv(path: Path):
    df = pd.read_csv(path)
    return df[['freq', 'Zreal', 'Zimag']]


def _load_mpt(path: Path):
    """非常简易的 BioLogic .mpt 解析：跳过非数据行直到出现 freq 或 frequency 列"""
    with open(path, 'r', errors='ignore') as f:
        lines = f.readlines()
    header_idx = None
    for idx, line in enumerate(lines):
        if re.search(r'freq|frequency', line, re.IGNORECASE):
            header_idx = idx
            break
    if header_idx is None:
        raise ValueError('Could not locate data header in .mpt file')
    df = pd.read_csv(path, skiprows=header_idx, sep='\t|,|;', engine='python')
    # 兼容不同列名
    freq_col = [c for c in df.columns if re.search(r'freq', c, re.IGNORECASE)][0]
    zreal_col = [c for c in df.columns if re.search(r'zreal|rez', c, re.IGNORECASE)][0]
    zimag_col = [c for c in df.columns if re.search(r'zimag|imz', c, re.IGNORECASE)][0]
    df = df.rename(columns={freq_col: 'freq', zreal_col: 'Zreal', zimag_col: 'Zimag'})
    return df[['freq', 'Zreal', 'Zimag']]


def _load_autolab_txt(path: Path):
    df = pd.read_csv(path, comment='#', delim_whitespace=True)
    freq_col = [c for c in df.columns if re.search(r'freq', c, re.IGNORECASE)][0]
    zreal_col = [c for c in df.columns if re.search(r'zreal|rez', c, re.IGNORECASE)][0]
    zimag_col = [c for c in df.columns if re.search(r'zimag|imz', c, re.IGNORECASE)][0]
    df = df.rename(columns={freq_col: 'freq', zreal_col: 'Zreal', zimag_col: 'Zimag'})
    return df[['freq', 'Zreal', 'Zimag']]


def load_eis(path: str | Path):
    """Detect file extension and load impedance data, returning freq and complex Z."""
    path = Path(path)
    ext = path.suffix.lower()
    if ext == '.csv':
        df = _load_csv(path)
    elif ext == '.mpt':
        df = _load_mpt(path)
    elif ext == '.txt':
        df = _load_autolab_txt(path)
    else:
        raise ValueError(f'Unsupported file extension: {ext}')
    freq = df['freq'].values
    z = df['Zreal'].values + 1j * df['Zimag'].values
    return freq, z 