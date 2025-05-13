###############################################################################
# Find CBLAS
#
# This sets the following variables:
# CBLAS_FOUND - True if CBLAS was found
# CBLAS_INCLUDE_DIRS - Header file directory
# CBLAS_LIBRARIES - Library to link against
###############################################################################

find_path(CBLAS_INCLUDE_DIR
  NAMES cblas.h
  PATHS
  /usr/include
  /usr/local/include
  /opt/homebrew/include
  /opt/homebrew/opt/openblas/include
  $ENV{CBLAS_HOME}/include
)

find_library(CBLAS_LIBRARY
  NAMES cblas
  PATHS
  /usr/lib
  /usr/local/lib
  /usr/lib/x86_64-linux-gnu
  /opt/homebrew/lib
  /opt/homebrew/opt/openblas/lib
  $ENV{CBLAS_HOME}/lib
)

set(CBLAS_LIBRARIES ${CBLAS_LIBRARY})
set(CBLAS_INCLUDE_DIRS ${CBLAS_INCLUDE_DIR})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CBLAS DEFAULT_MSG CBLAS_LIBRARY CBLAS_INCLUDE_DIR)

mark_as_advanced(CBLAS_INCLUDE_DIR CBLAS_LIBRARY) 