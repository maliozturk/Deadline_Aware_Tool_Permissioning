# =============================================================================
#  DEADLINE-AWARE TOOL PERMISSIONING (DATP)
#  Product Signature: DATP
# ------------------------------------------------------------------------------
#  File: General_Definitions/Types.py
#  Purpose: Define numpy type aliases used across the project.
#  Author: Muhammet Ali Ozturk
#  Generated: 2026-01-18
#  Environment: Python 3.9.13
# =============================================================================

                   

import numpy as np

INT8        :   type[np.int8]       = np.int8
INT16       :   type[np.int16]      = np.int16
INT32       :   type[np.int32]      = np.int32
INT64       :   type[np.int64]      = np.int64

UINT8       :   type[np.uint8]      = np.uint8
UINT16      :   type[np.uint16]     = np.uint16
UINT32      :   type[np.uint32]     = np.uint32
UINT64      :   type[np.uint64]     = np.uint64

FLOAT16     :   type[np.float16]    = np.float16
FLOAT32     :   type[np.float32]    = np.float32
FLOAT64     :   type[np.float64]    = np.float64

BYTE                                = UINT8
BOOL                                = np.bool_
