cmake_minimum_required(VERSION 3.5)

set(DROPLET_INSTALL "/usr/local/tangrt")
include(../cmake/FindDropletCompiler.cmake)


set(VENDOR_INCLUDE_DIRS ${VENDOR_INCLUDE_DIRS} ${DROPLET_INSTALL}/include PARENT_SCOPE)
set(VENDOR_LIB_DIRS ${VENDOR_LIB_DIRS} ${DROPLET_INSTALL}/lib/linux-x86_64 PARENT_SCOPE)


add_library(pccl SHARED pccl.cpp)
target_include_directories(pccl PUBLIC VENDOR_INCLUDE_DIRS)
target_include_directories(pccl PUBLIC .)
target_link_libraries(pccl PRIVATE VENDOR_LIB_DIRS)


# 将符号修改为弱符号的命令
add_custom_command(
    TARGET pccl
    POST_BUILD
    COMMAND ${CMAKE_OBJCOPY} --weaken-symbols="*" $<TARGET_FILE:pccl>
    COMMENT "Converting symbols to weak symbols in pccl library"
)
