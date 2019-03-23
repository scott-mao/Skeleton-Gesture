#include "hand.h"
using namespace cv;
hand::hand(cv::Size fSize,int fRate)
{
	frameSize = fSize;
	frameRate = fRate;
	cv::namedWindow("Depth", cv::WINDOW_AUTOSIZE);
	session = PXCSession::CreateInstance();
	new_frame= cv::Mat(fSize, CV_16UC1, Scalar(0));
	frame_rgb = cv::Mat(fSize, CV_8UC3, Scalar(0));
	joint_world.resize(66);
	pxcSenseManger = session->CreateSenseManager();
	pxcSenseManger->EnableStream(PXCCapture::STREAM_TYPE_COLOR, 640, 480);
		if (pxcSenseManger == NULL) {
			std::cout << "Unable to create the PXCSenseManager" << std::endl;
			pxcSenseManger->Release();
			session->Release();
			exit(0);
		}
		
	version = pxcSenseManger->QuerySession()->QueryVersion();
	std::cout << "SDK Version:" << version.major << version.minor << std::endl;

	cap = pxcSenseManger->QueryCaptureManager();
	
	// Enable hand tracking
	pxcSenseManger->EnableHand();
	// Get an instance of PXCHandModule 
	
	handModule = pxcSenseManger->QueryHand();
	// Get an instance of PXCHandConfiguration

	handConfig = handModule->CreateActiveConfiguration();
	//handConfig ->SetTrackingMode(PXCHandData::TRACKING_MODE_EXTREMITIES);
	handConfig->SetTrackingMode(PXCHandData::TRACKING_MODE_FULL_HAND);
	

	handm = handModule->CreateOutput();

	if (pxcSenseManger->Init() != PXC_STATUS_NO_ERROR)
	{
		std::cout << "Unable to Initialize the PXCSenseManager" << std::endl;
		exit(0);
	}	
};

hand::~hand()
{
	
	joint_world.~vector();
	//cap->Release();
	//pxcSenseManger->Release();
	session->Release();
	cvDestroyAllWindows();
	std::cout << "delete hand object" << std::endl;
}

void hand::frameRelease()
{
	pxcSenseManger->ReleaseFrame();
};

void hand::frame()
{
	new_frame = PXCImage2CVMat(sample->depth, PXCImage::PIXEL_FORMAT_DEPTH);
	cv::imshow("Depth", new_frame * 50);
	cv::waitKey(1);

};

void hand::RGBf()
{
	frame_rgb = PXCImage2CVMat(sample->color, PXCImage::PIXEL_FORMAT_RGB24);
	cv::imshow("RGB", frame_rgb);
	cv::waitKey(1);

};



Mat *hand::now_frame()
{
	return &new_frame;
}

Mat *hand::now_rgb()
{
	return &frame_rgb;
}

Mat hand::PXCImage2CVMat(PXCImage *pxcImage, PXCImage::PixelFormat format)
{
	PXCImage::ImageData data;

	pxcImage->AcquireAccess(PXCImage::ACCESS_READ, format, &data);

	int width = pxcImage->QueryInfo().width;
	int height = pxcImage->QueryInfo().height;

	if (!format)
		format = pxcImage->QueryInfo().format;

	int type;
	if (format == PXCImage::PIXEL_FORMAT_Y8)
		type = CV_8UC1;
	else if (format == PXCImage::PIXEL_FORMAT_RGB24)
		type = CV_8UC3;
	else if (format == PXCImage::PIXEL_FORMAT_DEPTH)
		type = CV_16UC1;
	//else if (format == PXCImage::PIXEL_FORMAT_DEPTH_F32)
	//	type = CV_32FC1;

	cv::Mat ocvImage = cv::Mat(cv::Size(width, height), type, data.planes[0]);

	pxcImage->ReleaseAccess(&data);
	return ocvImage;
};



void hand::updatedata()
{
	pxcSenseManger->AcquireFrame();
	handm->Update();
	sample = pxcSenseManger->QuerySample(); 
	
};

pxcI32 hand::gethandnum()
{
	return  handm->QueryNumberOfHands();
};

std::vector<float> *hand::joint_ptr()
{
	
	return &joint_world;
}

void hand::getworld()
{
	PXCHandData::IHand *hd;
	handm->QueryHandData(PXCHandData::ACCESS_ORDER_NEAR_TO_FAR, 0, hd);
	PXCHandData::JointData joints;
	PXCHandData::JointType index;
	int num=0;
	for (int i = 0; i < 22; i++)
	{
		if (i == 0)// Swap Wrist to 2nd , Center to 1st, since the order in JointType is Wrist 1st
			num = 3;
		else if(i == 1)
		    num = 0;
		else
			num = i * 3;

		hd->QueryTrackedJoint(static_cast<PXCHandData::JointType>(i), joints);
		joint_world[0 + num] = joints.positionWorld.x;
		joint_world[1 + num] = joints.positionWorld.y; 
		joint_world[2 + num] = joints.positionWorld.z;

	}
}

void hand::getworld_wrist1st()
{
	PXCHandData::IHand *hd;
	handm->QueryHandData(PXCHandData::ACCESS_ORDER_NEAR_TO_FAR, 0, hd);
	PXCHandData::JointData joints;
	PXCHandData::JointType index;
	int num = 0;
	for (int i = 0; i < 22; i++)
	{
		num = i * 3;

		hd->QueryTrackedJoint(static_cast<PXCHandData::JointType>(i), joints);
		joint_world[0 + num] = joints.positionWorld.x;
		joint_world[1 + num] = joints.positionWorld.y;
		joint_world[2 + num] = joints.positionWorld.z;

	}
}

void hand::getworld_normalize()
{
	PXCHandData::IHand *hd;
	handm->QueryHandData(PXCHandData::ACCESS_ORDER_NEAR_TO_FAR, 0, hd);
	PXCHandData::JointData joints;
	PXCHandData::JointType index;
	int num = 0;
	//Put center to 1st in joint_world
	hd->QueryTrackedJoint(static_cast<PXCHandData::JointType>(1), joints);
	joint_world[0 + num] = joints.positionWorld.x;
	joint_world[1 + num] = joints.positionWorld.y;
	joint_world[2 + num] = joints.positionWorld.z;

	float X_center= joints.positionWorld.x;
	float Y_center= joints.positionWorld.y;
	float Z_center= joints.positionWorld.z;

	// Put wrist to 2nd in joint_world
	num = 3;
	hd->QueryTrackedJoint(static_cast<PXCHandData::JointType>(0), joints);//
	joint_world[0 + num] = joints.positionWorld.x - X_center;
	joint_world[1 + num] = joints.positionWorld.y - Y_center;
	joint_world[2 + num] = joints.positionWorld.z - Z_center;

	for (int i = 2; i < 22; i++)
	{   
		
		num = i * 3;

		hd->QueryTrackedJoint(static_cast<PXCHandData::JointType>(i), joints);
		joint_world[0 + num] = joints.positionWorld.x - X_center;
		joint_world[1 + num] = joints.positionWorld.y - Y_center;
		joint_world[2 + num] = joints.positionWorld.z - Z_center;

	}

	
}
