function(opensn_get_caliper_version OUTVAR)
  set(_caliper_version "")
  if(DEFINED caliper_INCLUDE_DIR AND EXISTS "${caliper_INCLUDE_DIR}/caliper/caliper-config.h")
    file(STRINGS
      "${caliper_INCLUDE_DIR}/caliper/caliper-config.h"
      _caliper_version_line
      REGEX "^[ \t]*#define[ \t]+CALIPER_VERSION[ \t]+\"[0-9]+\\.[0-9]+\\.[0-9]+\"")
    if(_caliper_version_line MATCHES "\"([0-9]+\\.[0-9]+\\.[0-9]+)\"")
      set(_caliper_version "${CMAKE_MATCH_1}")
    endif()
  endif()
  set(${OUTVAR} "${_caliper_version}" PARENT_SCOPE)
endfunction()

function(opensn_check_caliper_version REQUESTED_VERSION OUTVAR)
  opensn_get_caliper_version(_caliper_version)
  if(_caliper_version STREQUAL "" OR _caliper_version VERSION_LESS REQUESTED_VERSION)
    set(${OUTVAR} FALSE PARENT_SCOPE)
  else()
    set(${OUTVAR} TRUE PARENT_SCOPE)
  endif()
endfunction()
