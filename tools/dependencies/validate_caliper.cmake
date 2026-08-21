set(CALIPER_CONFIG_HEADER
    "${CMAKE_INSTALL_PREFIX}/include/caliper/caliper-config.h")
if(NOT EXISTS "${CALIPER_CONFIG_HEADER}")
  message(FATAL_ERROR
          "Caliper validation failed: ${CALIPER_CONFIG_HEADER} is missing")
endif()

file(READ "${CALIPER_CONFIG_HEADER}" CALIPER_CONFIG)

function(require_caliper_feature FEATURE REQUIRED)
  string(FIND "${CALIPER_CONFIG}" "#define CALIPER_HAVE_${FEATURE}"
         FEATURE_POSITION)
  if(REQUIRED AND FEATURE_POSITION EQUAL -1)
    message(FATAL_ERROR
            "Caliper validation failed: CALIPER_HAVE_${FEATURE} is not enabled")
  endif()
endfunction()

require_caliper_feature(MPI ON)
require_caliper_feature(NVTX "${REQUIRE_NVTX}")
require_caliper_feature(CUPTI "${REQUIRE_CUPTI}")
require_caliper_feature(ROCPROFILER "${REQUIRE_ROCPROFILER}")

foreach(TOOL cali-query cali-stat mpi-caliquery cali2traceevent)
  if(NOT EXISTS "${CMAKE_INSTALL_PREFIX}/bin/${TOOL}")
    message(FATAL_ERROR
            "Caliper validation failed: ${TOOL} was not installed")
  endif()
endforeach()

execute_process(
  COMMAND "${CMAKE_INSTALL_PREFIX}/bin/cali-query" --help services
  RESULT_VARIABLE SERVICES_RESULT
  OUTPUT_VARIABLE CALIPER_SERVICES
  ERROR_VARIABLE SERVICES_ERROR)
if(NOT SERVICES_RESULT EQUAL 0)
  message(FATAL_ERROR
          "Caliper service query failed: ${SERVICES_ERROR}")
endif()

set(REQUIRED_SERVICES mpi mpireport)
if(REQUIRE_NVTX)
  list(APPEND REQUIRED_SERVICES nvtx)
endif()
if(REQUIRE_CUPTI)
  list(APPEND REQUIRED_SERVICES cupti cuptitrace)
endif()
if(REQUIRE_ROCPROFILER)
  list(APPEND REQUIRED_SERVICES rocprofiler roctracer roctx)
endif()
foreach(SERVICE IN LISTS REQUIRED_SERVICES)
  if(NOT CALIPER_SERVICES MATCHES "(^|\n)[ \t]*${SERVICE}[ \t]")
    message(FATAL_ERROR
            "Caliper validation failed: service '${SERVICE}' is not registered")
  endif()
endforeach()

if(REQUIRE_CUPTI OR REQUIRE_ROCPROFILER)
  execute_process(
    COMMAND "${CMAKE_INSTALL_PREFIX}/bin/cali-query" --help configs
    RESULT_VARIABLE CONFIGS_RESULT
    OUTPUT_VARIABLE CALIPER_CONFIGS
    ERROR_VARIABLE CONFIGS_ERROR)
  if(NOT CONFIGS_RESULT EQUAL 0)
    message(FATAL_ERROR
            "Caliper configuration query failed: ${CONFIGS_ERROR}")
  endif()
  set(REQUIRED_CONFIGS)
  if(REQUIRE_CUPTI)
    list(APPEND REQUIRED_CONFIGS cuda-activity-profile cuda-activity-report)
  endif()
  if(REQUIRE_ROCPROFILER)
    list(APPEND REQUIRED_CONFIGS rocm-activity-profile rocm-activity-report)
  endif()
  foreach(CONFIG IN LISTS REQUIRED_CONFIGS)
    if(NOT CALIPER_CONFIGS MATCHES "(^|\n)[ \t]*${CONFIG}[ \t]")
      message(FATAL_ERROR
              "Caliper validation failed: configuration '${CONFIG}' is unavailable")
    endif()
  endforeach()
endif()

message(STATUS
        "Validated Caliper MPI support and ${OPENSN_CALIPER_GPU_BACKEND} profiling features")
