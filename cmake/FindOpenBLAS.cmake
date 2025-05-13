###############################################################################
# Find OpenBLAS
#
# This sets the following variables:
# OPENBLAS_FOUND - True if OpenBLAS was found
# OPENBLAS_INCLUDE_DIRS - Header file directory
# OPENBLAS_LIBRARIES - Library to link against
###############################################################################

find_path(OPENBLAS_INCLUDE_DIR
  NAMES cblas.h
  PATHS
  /usr/include
  /usr/local/include
  /opt/homebrew/include
  /opt/homebrew/opt/openblas/include
  $ENV{OPENBLAS_HOME}/include
)

find_library(OPENBLAS_LIBRARY
  NAMES openblas
  PATHS
  /usr/lib
  /usr/local/lib
  /opt/homebrew/lib
  /opt/homebrew/opt/openblas/lib
  $ENV{OPENBLAS_HOME}/lib
)

set(OPENBLAS_LIBRARIES ${OPENBLAS_LIBRARY})
set(OPENBLAS_INCLUDE_DIRS ${OPENBLAS_INCLUDE_DIR})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(OpenBLAS DEFAULT_MSG OPENBLAS_LIBRARY OPENBLAS_INCLUDE_DIR)

mark_as_advanced(OPENBLAS_INCLUDE_DIR OPENBLAS_LIBRARY) 