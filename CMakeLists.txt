# CMakeLists.txt for Options Trading System C++ Engine

cmake_minimum_required(VERSION 3.12)
project(SIGOptionsEngine VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set build type
if(NOT CMAKE_BUILD_TYPE)
 set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler-specific options
if(MSVC)
 set(CMAKE_CXX_FLAGS_RELEASE "/O2 /DNDEBUG")
 set(CMAKE_CXX_FLAGS_DEBUG "/Od /Zi /DEBUG")
else()
 set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native")
 set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
endif()

# Enable OpenMP for parallel processing
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
 target_link_libraries(options_engine PUBLIC OpenMP::OpenMP_CXX)
 message(STATUS "OpenMP found and enabled")
endif()

# Include directories
include_directories(${PROJECT_SOURCE_DIR}/cpp_engine/include)

# Source files
set(SOURCES
 cpp_engine/src/math_utils.cpp
 cpp_engine/src/black_scholes.cpp
 cpp_engine/src/binomial_model.cpp
 cpp_engine/src/monte_carlo.cpp
 cpp_engine/src/volatility_surface.cpp
)

# Create the main library
add_library(options_engine STATIC ${SOURCES})

# Header files for installation
set(HEADERS
 cpp_engine/include/option_types.h
 cpp_engine/include/math_utils.h
 cpp_engine/include/black_scholes.h
 cpp_engine/include/binomial_model.h
 cpp_engine/include/monte_carlo.h
 cpp_engine/include/volatility_surface.h
)

# Python bindings using pybind11
find_package(pybind11 QUIET)
if(pybind11_FOUND)
 pybind11_add_module(sig_options_engine
 python_api/src/python_bindings.cpp
 ${SOURCES}
 )

 target_compile_definitions(sig_options_engine PRIVATE VERSION_INFO=${PROJECT_VERSION})
 message(STATUS "Python bindings enabled")
endif()

# Testing
enable_testing()

# Find Google Test
find_package(GTest QUIET)
if(GTest_FOUND)
 add_executable(options_engine_tests
 cpp_engine/tests/test_black_scholes.cpp
 cpp_engine/tests/test_binomial.cpp
 cpp_engine/tests/test_monte_carlo.cpp
 cpp_engine/tests/test_math_utils.cpp
 )

 target_link_libraries(options_engine_tests
 options_engine
 GTest::gtest_main
 )

 add_test(NAME AllTests COMMAND options_engine_tests)
 message(STATUS "C++ tests enabled")
endif()

# Benchmarking
option(BUILD_BENCHMARKS "Build benchmark tests" OFF)
if(BUILD_BENCHMARKS)
 find_package(benchmark QUIET)
 if(benchmark_FOUND)
 add_executable(options_benchmarks
 cpp_engine/benchmarks/pricing_benchmarks.cpp
 )

 target_link_libraries(options_benchmarks
 options_engine
 benchmark::benchmark
 )
 message(STATUS "Benchmarks enabled")
 endif()
endif()

# Documentation
find_package(Doxygen QUIET)
if(DOXYGEN_FOUND)
 set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/docs/Doxyfile.in)
 set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)

 configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)

 add_custom_target(doc_doxygen ALL
 COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
 WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
 COMMENT "Generating API documentation with Doxygen"
 VERBATIM
 )
 message(STATUS "Documentation generation enabled")
endif()

# Installation
install(TARGETS options_engine
 LIBRARY DESTINATION lib
 ARCHIVE DESTINATION lib
 RUNTIME DESTINATION bin
)

install(FILES ${HEADERS}
 DESTINATION include/sig_options
)

# Package configuration
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
 "${CMAKE_CURRENT_BINARY_DIR}/SIGOptionsEngineConfigVersion.cmake"
 VERSION ${PROJECT_VERSION}
 COMPATIBILITY AnyNewerVersion
)

# Print configuration summary
message(STATUS "")
message(STATUS "Configuration Summary:")
message(STATUS " Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS " C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS " Compiler: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS " Install prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "")

# Performance optimization flags for Release builds
if(CMAKE_BUILD_TYPE STREQUAL "Release")
 if(NOT MSVC)
 # GCC/Clang optimizations
 target_compile_options(options_engine PRIVATE
 -ffast-math
 -funroll-loops
 -finline-functions
 -flto
 )

 # Link-time optimization
 set_property(TARGET options_engine PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
 endif()

 message(STATUS "Release optimizations enabled")
endif()

# Compiler warnings
if(MSVC)
 target_compile_options(options_engine PRIVATE /W4)
else()
 target_compile_options(options_engine PRIVATE
 -Wall -Wextra -Wpedantic -Werror
 )
endif()

message(STATUS "Build configuration complete")