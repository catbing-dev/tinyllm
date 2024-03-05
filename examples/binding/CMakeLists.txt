set(TARGET binding)

add_library(${TARGET} SHARED binding.h binding.cpp)

target_include_directories(${TARGET} PUBLIC .)
target_include_directories(${TARGET} PUBLIC ../../src/)
target_compile_features   (${TARGET} PUBLIC cxx_std_11) # don't bump

target_link_libraries(${TARGET} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})

if (BUILD_SHARED_LIBS)
    set_target_properties(${TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(${TARGET} PRIVATE LLAMA_SHARED LLAMA_BUILD)
endif()