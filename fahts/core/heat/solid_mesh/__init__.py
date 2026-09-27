"""3-D Hex8 solid meshes for the FAHTS 3-D solver."""
from .solid_mesh import FACE_END, FACE_INNER, FACE_OUTER, SolidMesh
from .extrude import extrude_section
from .box_solid_mesher import BoxSolidMesher
from .iprofile_solid_mesher import IProfileSolidMesher
from .pipe_solid_mesher import PipeSolidMesher
from .plate_solid_mesher import PlateSolidMesher
from .block_solid_mesher import BlockSolidMesher

__all__ = [
    "SolidMesh", "FACE_OUTER", "FACE_INNER", "FACE_END",
    "extrude_section",
    "BoxSolidMesher", "IProfileSolidMesher", "PipeSolidMesher",
    "PlateSolidMesher", "BlockSolidMesher",
]
