"""Spectra written as JCAMP-DX are read back like the NIST library files."""

from __future__ import annotations

import numpy as np

from esper.libraries import JDX_MAX_Y, Spectrum, read_jdx, write_jdx


def test_write_jdx_round_trip(tmp_path):
    # 11 peaks: several lines of the peak table; m/z 60 rounds to 0 and m/z 70 is empty
    mz = [31, 35, 47, 50, 51, 60, 69, 70, 85, 87, 101]
    intensities = [404.0, 20.0, 3.5, 76.0, 9999.0, 0.01, 5848.0, 0.0, 211.0, 158.0, 12.0]
    file = tmp_path / "241007.0423.air.1_CFC-12.jdx"
    write_jdx(file, Spectrum.from_pairs(mz, intensities), title="CFC-12", comments={"MEASUREMENT": "241007.0423.air.1"})

    text = file.read_text()
    assert "##TITLE=CFC-12\n" in text
    assert "##$MEASUREMENT=241007.0423.air.1\n" in text
    assert "##NPOINTS=9\n" in text
    assert text.endswith("##END=\n")

    spectrum = read_jdx(file)
    expected = [m for m, i in zip(mz, intensities) if round(JDX_MAX_Y * i / 9999.0) > 0]
    np.testing.assert_array_equal(spectrum.mz, expected)
    assert spectrum.intensities.max() == JDX_MAX_Y
    np.testing.assert_allclose(spectrum.intensities[spectrum.mz == 69], 5848)
