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
`Different download versions may result in a chance of missing or adding libraries. Please check it yourself` \
`The project comes with 5 pre-trained models `\
`shape, which is an object detection model for detecting the positions of basic shapes in images `\
`resShape, which is an image classification model for classifying basic shapes `\
`carcard, which is an object detection model for detecting the positions of license plates `\
`main, which is an object detection model with 11 classes `\
`and color, which is an image classification model for classifying colors `\
```testCode
//we can use ResNet or Yolo to create model.
ResNet model;
//put modelPath to Init
model.Init("./model/color");
//we can use "utils::Dectet" to dectet image and video and file
//Dectet(string path, Model* model, vector<string> classes, bool saveFlag, string savePath, bool showFlag)
//saveFlag is set to true by default, which means the processed image or video will be saved. The default save location is the "output" folder in this project. showFlag is set to false by default, which means the processed image will not be displayed.
utils::Dectet("./images", &model, utils::colorClasses);
```
## If you want to join your own new model,
specifically a YOLOv5 model, you need to make the following modifications in the .param file: \
1. In the line with three Permute operations, change the fourth parameter to "output", "output1", and "output2" respectively. \
2. Each Permute operation has a corresponding reshape operation. In the line with the reshape operation, change the fifth parameter from 0=? to 0=-1. \
For classification models, modify the .param file as follows:\
1. In the first line with the Input operation, change the three parameters to 1 1 images.\
2. In the second line, change the first three parameters to 1 1 images. \
3. In the line with the InnerProduct operation, change the fourth parameter to "output". \
By making these modifications, you will be able to run the model within the current framework. It is recommended to use YOLOv5-5.6.2 for converting to ONNX and NCNN. The YOLOv5s, YOLOv5m, and YOLOv5s6 models may not require any additional operations and can be used directly. \


