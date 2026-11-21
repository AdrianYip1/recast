"""One module per shape mode. Each returns a trimesh.Trimesh in the object photo's world frame."""

from .box import build_box
from .card import build_card
from .extrude import build_extrude
from .revolution import build_revolution

__all__ = ["build_box", "build_card", "build_extrude", "build_revolution"]
