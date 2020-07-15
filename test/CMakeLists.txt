cmake_minimum_required(VERSION 3.5)
project(tests)

option(WITH_HDF5 "use HDF5" ON)
option(WITH_UNITTEST "builds additional unit tests" ON)
option(WITH_MPI "use MPI" ON)

set(LOG_LEVEL "info" CACHE STRING "Log level for the code")
set(LOG_LEVEL_OPTIONS "debug" "info" "warning" "error")
set_property(CACHE LOG_LEVEL PROPERTY STRINGS ${LOG_LEVEL_OPTIONS})

if (${LOG_LEVEL} STREQUAL "debug")
    set(DEFS "LOGLEVEL=3")
elseif (${LOG_LEVEL} STREQUAL "info")
    set(DEFS "LOGLEVEL=2")
elseif (${LOG_LEVEL} STREQUAL "warning")
    set(DEFS "LOGLEVEL=1")
elseif (${LOG_LEVEL} STREQUAL "error")
    set(DEFS "LOGLEVEL=0")
endif()

if (WITH_UNITTEST)
    # NOTE: Unittest always require MPI (see travis)
    #       Additionally, unittest cannot get compiled without mpi
    set(WITH_MPI ON)
endif()

if (WITH_MPI)
    find_package(MPI REQUIRED)
    set(INCLUDES ${MPI_CXX_INCLUDE_DIRS})
    set(LIBS ${MPI_CXX_LIBRARIES})
    set(DEFS ${DEFS} USE_MPI)
endif()

if (WITH_HDF5)
    find_package(HDF5 REQUIRED)
    set(INCLUDES ${INCLUDES} ${HDF5_INCLUDE_DIRS})
    set(LIBS ${LIBS} ${HDF5_LIBRARIES})
    set(DEFS ${DEFS} USE_HDF5)
endif()

set(INCLUDES ${INCLUDES} ".." "../submodules")

add_executable(xdmfwriter src/xdmfwritertest.cpp)
target_link_libraries(xdmfwriter ${LIBS})
target_include_directories(xdmfwriter PUBLIC ${INCLUDES})
target_compile_definitions(xdmfwriter PUBLIC ${DEFS})


function(parallel_cxx_test target source mpi_entry_point)
    # adjust names a generated cpp file
    get_filename_component(CPP_FILE_NAME ${source} NAME)
    string(REGEX REPLACE "h$|hpp$" "cpp" CPP_FILE_NAME ${CPP_FILE_NAME})
    set(CPP_FULL_NAME "${CMAKE_CURRENT_BINARY_DIR}/${CPP_FILE_NAME}")

    add_custom_command(
            OUTPUT "${CPP_FULL_NAME}"
            COMMAND ${CXXTESTGEN} --runner=ErrorPrinter --main=cxxtest_main --output "${CPP_FULL_NAME}" "${source}"
            DEPENDS "${source}"
    )
    add_executable(${target} ${mpi_entry_point} ${CPP_FULL_NAME})
    set_target_properties(${target} PROPERTIES COMPILE_FLAGS "-Wno-effc++")

    # INCLUDES, LIBS and DEFS are given from the global scope defined above
    target_include_directories(${target} PUBLIC ${INCLUDES})
    target_compile_definitions(${target} PUBLIC ${DEFS})
    target_link_libraries(${target} ${LIBS})

    # the original project always ran with 3 MPI processes
    set (test_parameters -np 3 "./${target}")
    add_test(NAME ${target} COMMAND "mpirun" ${test_parameters})
endfunction(parallel_cxx_test)

if (WITH_UNITTEST)
    enable_testing()

    # add cxxtest from submodules to the project
    set(CXXTEST_PATH ../submodules/cxxtest)
    get_filename_component(CXXTEST_CMAKE "${CXXTEST_PATH}/build_tools/cmake/CxxTest.cmake" ABSOLUTE)
    include("${CXXTEST_CMAKE}")
    set(INCLUDES ${INCLUDES} ${CXXTEST_INCLUDES})

    # find all necessary files
    set(MPI_ENTRY_POINT ${CMAKE_SOURCE_DIR}/unittests/main.cpp)
    file(GLOB TESTFILES "unittests/*.t.h")

    # generate and add each test
    foreach(FILE ${TESTFILES})
        get_filename_component(TESTNAME ${FILE} NAME)
        string(REGEX REPLACE ".t.h$|.t.hpp$" "-Test" TESTNAME ${TESTNAME})
        parallel_cxx_test(${TESTNAME} ${FILE} ${MPI_ENTRY_POINT} )
    endforeach()
endif()
