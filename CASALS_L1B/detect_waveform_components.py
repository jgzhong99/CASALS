"""Compatibility wrapper for the renamed waveform feature extractor.

Use ``extract_waveform_features.py`` as the main batch entry point.
"""

from __future__ import annotations

from extract_waveform_features import main


if __name__ == "__main__":
    main()
