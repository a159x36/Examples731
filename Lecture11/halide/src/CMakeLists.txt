
#########################################################
# Source Files
#########################################################

add_executable(halide_generator halide_generator.cpp)
target_link_libraries(halide_generator PRIVATE Halide::Generator)

function(make_halide_target tgt triple)
    add_halide_library(${tgt}halide_dens_step FROM halide_generator GENERATOR halide_dens_step FUNCTION_NAME halide_dens_step TARGETS ${triple})
    add_halide_library(${tgt}halide_vel_step FROM halide_generator GENERATOR halide_vel_step FUNCTION_NAME halide_vel_step TARGETS ${triple})
    add_halide_library(${tgt}halide_bitmap FROM halide_generator  GENERATOR halide_bitmap FUNCTION_NAME halide_bitmap TARGETS ${triple})

    add_executable(${tgt}main main.cpp)
    target_link_libraries(${tgt}main ${tgt}halide_dens_step ${tgt}halide_vel_step ${tgt}halide_bitmap ${OpenCV_LIBS})
endfunction()

make_halide_target("" host)
make_halide_target(vulkan_ x86-64-linux-vulkan-profile)
make_halide_target(cuda_ host-cuda-profile)
make_halide_target(opencl_ host-opencl-profile)
make_halide_target(opengl_ host-openglcompute-profile)

