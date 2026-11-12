"""Processing of the chromatograms of a data directory: from the chromatograms to the
identified, fitted peaks (``esper peaks``, and ``esper upload`` for the runs imported
without peaks).

1. detection: the peaks of each mass, grouped into compounds (:mod:`.detection`);
2. identification, by the method of the data directory (:mod:`.identification`):
   ``target`` (target search) or ``suspect`` (suspect screening);
3. the peaks and compounds are stored as the extraction of the measurement
   (``measurement.peak_source`` :data:`PEAK_SOURCE`, ``measurement.identification``
   the method);
4. fit (:mod:`esper.fitting`, as for the extraction of alpinac).

Nothing depends on the instrument: the method and the parameters of each step are those
of the data directory (``processing:`` of its esper.yaml, :mod:`.config`). The steps
are in :mod:`.pipeline`; this module only names the methods (cheap to import). The runs
imported with peaks (alpinac's extraction and identification) are only fitted.
"""

PEAK_SOURCE = "esper"
"""``measurement.peak_source`` of the peaks found by esper (fitted: esper.fitting)."""
TARGET = "target"
SUSPECT = "suspect"
METHODS = (TARGET, SUSPECT)
"""How the compounds are identified: by the expected rt and ions of the targets (target
search: the instruments without full spectra, e.g. a quadrupole in SIM), or by their
spectrum in the libraries (suspect screening, as alpinac)."""
