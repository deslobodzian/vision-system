# For running flatc on *.fbs files

set(FLATBUFFERS_FLATC_EXECUTABLE flatc)

function(compile_fbs_for_target target) 
    # Schemas need to be in base directory as CMakeLists.txt
    message("Running FLATC")
    set(FBS_SCHEMA_DIR ${CMAKE_CURRENT_SOURCE_DIR}/schemas)
    set(FBS_GENERATED_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated_schemas)

    file(GLOB_RECURSE FBS_FILES ${FBS_SCHEMA_DIR}/*.fbs)
    file(MAKE_DIRECTORY ${FBS_GENERATED_DIR})

    message(${FBS_FILES})
    foreach(FBS_FILE ${FBS_FILES})
        message(${FBS_FILE})
        get_filename_component(FBS_NAME ${FBS_FILE} NAME_WE)
        set(GENERATED_FBS_SRC ${FBS_GENERATED_DIR}/${FBS_NAME}_generated.h)

        add_custom_command(
            OUTPUT ${GENERATED_FBS_SRC}
            COMMAND ${CMAKE_COMMAND} -E echo "Running flatc on ${FBS_FILE}"
            COMMAND ${FLATBUFFERS_FLATC_EXECUTABLE} -c -o ${FBS_GENERATED_DIR} ${FBS_FILE}
            COMMAND ${FLATBUFFERS_FLATC_EXECUTABLE} --python -o ${FBS_GENERATED_DIR} ${FBS_FILE}
            DEPENDS ${FBS_FILE}
            COMMENT "Compiling ${FBS_FILE} to ${GENERATED_FBS_SRC}"
        )
        list(APPEND GENERATED_FBS_SRCS ${GENERATED_FBS_SRC})
    endforeach()
    target_sources(${target} PRIVATE ${GENERATED_FBS_SRCS})
    target_include_directories(${target} PRIVATE ${FBS_GENERATED_DIR})
    add_dependencies(${target} GenerateFlatBuffers)

    add_custom_target(GenerateFlatBuffers DEPENDS ${GENERATED_FBS_SRCS})
endfunction()


