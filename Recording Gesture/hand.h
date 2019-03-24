#ifndef _HAND_HPP_
#define _HAND_HPP_
#include <pxcprojection.h>
#include <pxcsensemanager.h>
#include <pxcsession.h>
#include <pxchandconfiguration.h>
#include <pxchandmodule.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp> 
#include <vector>


// A class to manage the SDK session and to get hand information, in vector<float> form
class hand 
{
public:
	hand(cv::Size = cv::Size(640,480),int = 30);
	~hand();
	//Updata the data from new catching raw data,need to be called at every iteration's begining
	void updatedata();
	//get the world coordinate of joints and save it to the joint_world array, order: Center ->Wrist->Thumb->...->Pinky, totally 66 floating data(xyz) in a array
	void getworld();
	void getworld_wrist1st();
	//Normalize all data to be center at hand's center, except for hand's center itself with index 0~2
	void getworld_normalize();
	pxcI32 gethandnum();
	//Show new catching frame
	void frame();
	void RGBf();
	// Release the access of the data so it can be updataed with new data,need to be called at every iteration's ending
	void frameRelease();
	cv::Mat* now_frame();
	cv::Mat* now_rgb();
	//Return the pointer to the array of world coordinate of the joints
	std::vector<float> *joint_ptr();
	

private:
	std::vector<float>joint_world;
	cv::Size frameSize;
	int frameRate;
	PXCSession *session;
	PXCSenseManager *pxcSenseManger;
	PXCCaptureManager *cap;
	PXCHandModule *handModule;
	PXCHandConfiguration *handConfig;
	PXCCapture::Sample *sample;
	PXCSession::ImplVersion version;
	PXCHandData *handm;
    cv::Mat PXCImage2CVMat(PXCImage *pxcImage, PXCImage::PixelFormat format);
	cv::Mat new_frame;
	cv::Mat frame_rgb;
};




#endif