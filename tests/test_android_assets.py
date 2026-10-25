"""Explicit garment annotations must retain white fabric and reject bad inputs."""
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from prepare_android_assets import binary_annotation


def test_white_fabric_uses_annotation_independently_of_rgb():
    annotation = Image.new("L", (16, 20), 0)
    ImageDraw.Draw(annotation).rectangle((3, 3, 12, 17), fill=255)
    mask = binary_annotation(annotation, (16, 20))
    assert mask.dtype == np.bool_
    assert mask[8, 7]
    assert not mask[0, 0]


@pytest.mark.parametrize("size,fill", [((16, 20), 0), ((16, 20), 255), ((15, 20), 100)])
def test_empty_or_misaligned_annotation_is_rejected(size, fill):
    with pytest.raises(ValueError):
        binary_annotation(Image.new("L", size, fill), (16, 20))
