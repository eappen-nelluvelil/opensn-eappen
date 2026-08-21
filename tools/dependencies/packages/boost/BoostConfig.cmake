# Header-only Boost package configuration for the OpenSn dependency bundle.
get_filename_component(_BOOST_PREFIX "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)

set(Boost_FOUND TRUE)
set(Boost_VERSION 1.86.0)
set(Boost_VERSION_STRING 1.86.0)
set(Boost_INCLUDE_DIR "${_BOOST_PREFIX}/include")
set(Boost_INCLUDE_DIRS "${Boost_INCLUDE_DIR}")

if(NOT TARGET Boost::headers)
  add_library(Boost::headers INTERFACE IMPORTED)
  set_target_properties(Boost::headers PROPERTIES
                        INTERFACE_INCLUDE_DIRECTORIES "${Boost_INCLUDE_DIR}")
endif()
if(NOT TARGET Boost::boost)
  add_library(Boost::boost INTERFACE IMPORTED)
  set_target_properties(Boost::boost PROPERTIES
                        INTERFACE_LINK_LIBRARIES Boost::headers)
endif()

set(Boost_LIBRARIES Boost::headers)
unset(_BOOST_PREFIX)
