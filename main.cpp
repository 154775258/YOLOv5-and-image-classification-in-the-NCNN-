#include"model.h"
#include"utils.h"

int main() {
	//utils::outImageTest("D:\\pytorch\\data\\shapeDectet\\train\\images", "./model/shape");
	ResNet shape;
	shape.Init("./model/color");
	utils::Dectet("./images", &shape, utils::colorClasses);
}