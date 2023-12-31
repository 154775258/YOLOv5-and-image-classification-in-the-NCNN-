#include"model.h"
#include"utils.h"

int main() {
	//we can use ResNet or Yolo to create model.
	ResNet model;
	//put modelPath to Init
	model.Init("./model/color");
	//we can use "utils::Dectet" to dectet image and video and file
	//Dectet(string path, Model* model, vector<string> classes, bool saveFlag, string savePath, bool showFlag)
	//saveFlag is set to true by default, which means the processed image or video will be saved. The default save location is the "output" folder in this project. showFlag is set to false by default, which means the processed image will not be displayed.
	utils::Dectet("./images", &model, utils::colorClasses);
}