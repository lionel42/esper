"""Small synthetic TOFWERK h5 files (as in the tests of twutils).

- :func:`write_h5`: one compound with two mass peaks; with ``steps``, the TPS2 channels of
  the acquisition steps (filament emission, resonant ejection of m/z 100);
- :func:`write_mc`: a calibration file of the legacy alpinac.
"""

from __future__ import annotations

import h5py
import numpy as np
import pytest
from masscal import mass_to_tof

N_WRITES, N_BUFS, N_POINTS = 40, 10, 6000
S_PER_BIN = 0.2
CAL = {"p1": 600.0, "p2": -200.0, "p3": 0.5}  # power mode (TOFWERK 2)
RT_APEX = 40.0
PEAKS = {(69.0, 1000.0), (100.0, 400.0)}  # (m/z, height)
SIGMA_B, SIGMA_SLOPE, ALPHA = 0.5, 4e-5, 0.5


def pseudo_voigt(x, mu, sigma, alpha):
    gauss = np.exp(-((x - mu) ** 2) / (2 * sigma) ** 2)  # legacy convention
    lorentz = 1 / (1 + ((x - mu) / sigma) ** 2)
    return (1 - alpha) * gauss + alpha * lorentz


def write_timing(f: h5py.File, buf_times: np.ndarray, period: float, unwritten: int = 0, attrs: bool = True):
    """``TimingData`` (``unwritten`` last buffers at time 0) and the ``NbrWrites`` / ``NbrBufs`` attributes."""
    n_writes, n_bufs = buf_times.shape
    buf_times = buf_times.copy()
    if unwritten:
        buf_times.ravel()[-unwritten:] = 0
    td = f.create_group("TimingData")
    td.create_dataset("BufTimes", data=buf_times, chunks=(1, 1))
    td.attrs["BlockPeriod"] = np.array([period * 1e9])
    if attrs:  # complete writes only
        f.attrs["NbrWrites"] = np.array([n_writes - int(np.ceil(unwritten / n_bufs))], dtype=np.int32)
        f.attrs["NbrBufs"] = np.array([n_bufs], dtype=np.int32)
        f.attrs["NbrBlocks"] = np.array([1], dtype=np.int32)


STEP_EMISSION, STEP_EJECTION = 20.0, (50.0, 70.0)  # s
STEP_CHANNELS = ["FILAMENT_EMISSION monitor [mA]", "FILAMENT_EMISSION target [mA]", "QUADA_REXAMPL1 target [V]",
                 "QUADA_REXMASSE1 target [Th]"]


def write_h5(path, n_segments: int = 1, unwritten: int = 0, attrs: bool = True, steps: bool = False):
    """Synthetic run. With ``steps``, the acquisition steps of the real runs: the emission target
    0.05 -> 0.5 mA at STEP_EMISSION (background x10), the ejection of m/z 100 during STEP_EJECTION
    (background / 50); logged in TPS2 0.4 s after they take effect."""
    rng = np.random.default_rng(0)
    rt = (np.arange(N_WRITES * N_BUFS) + 1) * S_PER_BIN
    tof = np.arange(N_POINTS, dtype=float)
    data = rng.normal(1.0, 0.3, (len(rt), N_POINTS)).astype(np.float32)
    if steps:
        data[rt < STEP_EMISSION] /= 10
        ejected = (rt >= STEP_EJECTION[0]) & (rt < STEP_EJECTION[1])
        window = (tof > mass_to_tof(99.5, CAL, "power")) & (tof < mass_to_tof(100.5, CAL, "power"))
        data[np.ix_(ejected, window)] /= 50
    chromato = np.exp(-((rt - RT_APEX) ** 2) / (2 * 1.5**2))
    for mz, height in PEAKS:
        mu = float(mass_to_tof(mz, CAL, "power"))
        shape = pseudo_voigt(tof, mu, SIGMA_B + SIGMA_SLOPE * mu, ALPHA)
        data += (height * chromato[:, None] * shape).astype(np.float32)
    if unwritten:
        data[-unwritten:] = 0
    data = data.reshape(N_WRITES, N_BUFS, 1, N_POINTS)
    data = np.repeat(data, n_segments, axis=2) / n_segments
    with h5py.File(path, "w") as f:
        fs = f.create_group("FullSpectra")
        fs.create_dataset("TofData", data=data, chunks=(1, 1, 1, N_POINTS))
        fs.create_dataset("MassAxis", data=((tof - CAL["p2"]) / CAL["p1"]) ** 2)
        fs.attrs["MassCalibMode"] = np.array([2], dtype=np.int32)
        for k, v in CAL.items():
            fs.attrs[f"MassCalibration {k}"] = np.array([v])
        write_timing(f, rt.reshape(N_WRITES, N_BUFS), S_PER_BIN, unwritten, attrs)
        if steps:
            logged = rt - 0.4
            tps = np.zeros((len(rt), len(STEP_CHANNELS)))
            tps[:, 1] = np.where(logged < STEP_EMISSION, 0.05, 0.5)
            tps[:, 0] = tps[:, 1] * 0.98
            tps[:, 2] = np.where((logged >= STEP_EJECTION[0]) & (logged < STEP_EJECTION[1]), 3.0, 0.0)
            tps[:, 3] = 97.0
            f.create_dataset("TPS2/TwData", data=tps.reshape(N_WRITES, N_BUFS, -1))
            f.create_dataset("TPS2/TwInfo", data=np.array([c.encode() for c in STEP_CHANNELS], dtype="S256"))
    return path


def write_mc(path, rts=(0.0, 400.0)):
    """Calibration file of the legacy alpinac, same calibration at every rt."""
    lines = [f"{len(rts)}\tNumber of points, mass cal. vs RT", "2\tNumber of points, mass cal. u vs mass",
             "2\tMass cal. mode", f"{SIGMA_SLOPE}\tPeak shape, sigma_slope vs TOF index",
             f"{SIGMA_B}\tPeak shape, sigma_b vs TOF index", "8e-05\tPeak shape, sigma_slope vs mass",
             "0.0015\tPeak shape, sigma_b vs mass", f"{ALPHA}\talpha, for tofidx Pseudo-Voigt fit",
             f"{S_PER_BIN}\ts_per_bin"]
    lines += [f"{int(rt / S_PER_BIN)}\t{CAL['p1']}\t{CAL['p2']}\t{CAL['p3']}" for rt in rts]
    lines += ["68.9946609\t2.50", "130.9914672\t1.50"]
    path.write_text("\n".join(lines) + "\n")
    return path


@pytest.fixture
def h5_file(tmp_path):
    path = write_h5(tmp_path / "241007.0423.air.1.h5")
    (tmp_path / "241007.0423.air.1.alpinac").mkdir()
    write_mc(tmp_path / "241007.0423.air.1.alpinac" / "241007.0423.air.1.EI_mc.txt")
    return path
