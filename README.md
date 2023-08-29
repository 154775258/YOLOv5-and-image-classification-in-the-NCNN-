# YOLOv5-and-image-classification-in-the-NCNN
## first we need install package
Vulkan https://developer.nvidia.com/vulkan-driver \
ncnn  https://github.com/Tencent/ncnn/releases install ncnn-20230816-windows-vs2022.zip\
opencv https://opencv.org/releases/ opencv4.7.0\

## second open ncnn.sln
navigate to "Configuration Properties" -> "C/C++" -> "General". Add the path to the "Include Directories" field \
"yourPath"\Vulkan\Include \
"yourPath"\ncnn-20230816-windows-vs2022\x64\include \
"yourPath"\opencv\opencv\build\include \

In the same Properties window, "Library" field \
"yourPath"\opencv\opencv\build\x64\vc16\lib \
"yourPath"\Vulkan\Lib \
"yourPath"\ncnn-20230816-windows-vs2022\x64\lib \

navigate to "Configuration Properties" -> "Linker" -> "Input"\
dxcompiler.lib\
GenericCodeGen.lib\
glslang-default-resource-limits.lib\
glslang.lib\
HLSL.lib\
MachineIndependent.lib\
OGLCompiler.lib\
OSDependent.lib\
shaderc.lib\
shaderc_combined.lib\
shaderc_shared.lib\
shaderc_util.lib\
spirv-cross-c-shared.lib\
spirv-cross-c.lib\
spirv-cross-core.lib\
spirv-cross-cpp.lib\
spirv-cross-glsl.lib\
spirv-cross-hlsl.lib\
spirv-cross-msl.lib\
spirv-cross-reflect.lib\
spirv-cross-util.lib\
SPIRV-Tools-diff.lib\
SPIRV-Tools-link.lib\
SPIRV-Tools-lint.lib\
SPIRV-Tools-opt.lib\
SPIRV-Tools-reduce.lib\
SPIRV-Tools-shared.lib\
SPIRV-Tools.lib\
SPIRV.lib\
SPVRemapper.lib\
vulkan-1.lib\
ncnn.lib\
opencv_world470.lib\
opencv_world470d.lib\

Different download versions may result in a chance of missing or adding libraries. Please check it yourself、
