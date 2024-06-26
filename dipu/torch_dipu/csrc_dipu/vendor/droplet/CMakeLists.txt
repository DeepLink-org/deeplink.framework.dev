cmake_minimum_required(VERSION 3.5)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

set(DROPLET_INSTALL "/usr/local/tangrt")
include(cmake/FindDropletCompiler.cmake)

set(USE_PCCL "AUTO" CACHE STRING "Whether to compile with or without PCCL. AUTO(default): auto-detecting; 1|ON|YES|TRUE|Y: force compiling with PCCL; 0|OFF|NO|FALSE|N: force compiling without PCCL")

find_package(PCCL)
if (USE_PCCL)
    if ("${USE_PCCL}" AND NOT PCCL_FOUND)
        message(FATAL_ERROR "Set to force compiling with PCCL, but PCCL is not found.")
    endif()
    if(PCCL_FOUND)
        message(STATUS "Use PCCL to implement communicator")
        message(STATUS "PCCL_LIBRARY: " ${PCCL_LIBRARY})
        message(STATUS "PCCL_INCLUDE_DIR: " ${PCCL_INCLUDE_DIR})
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_PCCL" PARENT_SCOPE)
        set(DIPU_VENDOR_LIB ${DIPU_VENDOR_LIB} ${PCCL_LIBRARY})
        set(VENDOR_INCLUDE_DIRS ${VENDOR_INCLUDE_DIRS} ${PCCL_INCLUDE_DIR})
    else()
        message(STATUS "PCCL not found, i.e. not supporting distributed on droplet")
    endif()
else()
    message(STATUS "Force compiling without PCCL, i.e. not supporting distributed on droplet")
endif()


set(VENDOR_INCLUDE_DIRS ${VENDOR_INCLUDE_DIRS} ${DROPLET_INSTALL}/include PARENT_SCOPE)
set(VENDOR_LIB_DIRS ${VENDOR_LIB_DIRS} ${DROPLET_INSTALL}/lib/linux-x86_64 PARENT_SCOPE)
set(DIPU_VENDOR_LIB ${DIPU_VENDOR_LIB} tangrt_shared PARENT_SCOPE)

file(GLOB SRC_FILES  *.cpp)
set(VENDOR_FILES  ${SRC_FILES} PARENT_SCOPE)
