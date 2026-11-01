
cmake_minimum_required(VERSION 3.26)

project("159731_Project" CXX C)

#########################################################
# Find OpenCV
#########################################################
# Try to find Major version 5 first
find_package(OpenCV 5 QUIET)

# If version 5 isn't found, fall back to version 4.8
if(NOT OpenCV_FOUND)
    find_package(OpenCV 4.8 REQUIRED)
endif()

#########################################################
# Include directories
#########################################################
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories("${PROJECT_SOURCE_DIR}/src")
include_directories("${PROJECT_SOURCE_DIR}/include")

#########################################################
# Set Compiler Flags
#########################################################

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
	add_compile_options(/std:c++latest /utf-8 /W4 /MP /wd4800)
else()
	add_compile_options("$<$<NOT:$<CONFIG:Debug>>:-O2>")
	add_compile_options(-std=c++17 -Wall -Wextra -pedantic)
	add_compile_options(-fvisibility=hidden -pthread)
	# Uncomment for profiling
	#add_compile_options(-pg)
	#add_link_options(-pg)
	add_compile_options(-Werror=return-type)
endif()

#########################################################
# Source Files
#########################################################

add_subdirectory(src) # Primary source files
