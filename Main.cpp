/*
This programe will recode the depth data of each frame into a single png 16bits file
, and will also recode the frame at which the hand is detected. And start from the 
time the hand is detected , it will capture the joint information.

The loop will not end until pressing 'a' button or leaving the hand out of the deteting area of the camera, and then output all data into 2 "XXX.csv" files
all files will be saved at sample/fileX  folder, while X is the word you type at the begining.

Pressing any button except for 'a' will record the frame of the starting or ending of the gesture into the "section.csv"
file.

Chi-Tsung, Chang  2017/9/21
*/

#define _SCL_SECURE_NO_WARNINGS
#define GLOG_NO_ABBREVIATED_SEVERITIES
#define USE_OPENCV

#include <iostream>
#include <vector>
#include <fstream>
#include "hand.h"
#include "parameter.h"
#include <windows.h>
#include <thread>
#include "assist.h"

using namespace std;


int main()
{   
	string command = "mkdir ";
	string filename;
	cout << "Please input file number: ";
	cin >> filename;
	filename.insert(0, "sample\\file");//folder name for each sample
	command += filename;
	const char*cptr = command.c_str();
	system(cptr);//depth data will be sent to this dir

	//Create object
	hand myhand(frameSize, frameRate);
	vector<float> *joint = myhand.joint_ptr();
	std::vector<vector<float>> newdata;
	bool detect_switch = false;
	
	string K = filename + "\\joint.csv";
	//define file name
	ofstream out(K);//use for saving joint's data
	
	K = filename + "\\spec.csv";
	//use for saving the start of hand-detected-frame , overall (hand-detected)frames,and labels  number
	ofstream out2(K);
	int counter = 0;//counter 
	int maincounter = 0;
	int labelnum = 0;
	bool stop=false;
	thread mythread(Record,&stop,filename,&counter,&labelnum);
	command = "mkdir ";

	K = filename + "\\data";
	command += K;
	system(cptr);
	K += "\\hi";
	while(1)
	{
		
		myhand.updatedata();
		//Save depth map
		//myhand.frame();
		//cv::imwrite(K+std::to_string(maincounter)+".png", *myhand.now_frame());//saving file
		//*************
		
		//Save RGB data
		myhand.RGBf();
		cv::imwrite(K + std::to_string(maincounter) + "rgb.png", *myhand.now_rgb());//saving file
		//**************
		if (myhand.gethandnum() > 0)
		{
			if (detect_switch == false)
			{
				cout << "\nDetected hand, start recording." << endl;
				out2 << maincounter << endl;//saving the frame
				detect_switch = true;// will disable the continuous saving 
			}

			myhand.getworld_wrist1st();
			newdata.push_back(*joint);
			counter++;
		}
		//else if (myhand.gethandnum() > 0 && detect_switch)
			//detect_switch = false;//will make next new-handed-detection' frame be recordable

		else if (myhand.gethandnum() == 0 && detect_switch && counter > 0)
		{
			stop = true;//Stop loop when the hand leave the detecting area of the camera
			
		}
		
		myhand.frameRelease();
		maincounter++;
		if (stop)
		{
			
			std::cout << "stop manually " << endl;
			break;
		}
		
		
	}

	int row = newdata.size();
	int column = newdata[0].size();
	out2 << row << endl;
	cout << "Totally recoded hand gesture:"<<labelnum<<endl;
	out2 << labelnum;
	std::cout << "Frame's number with hand-detected:"<<row << endl;
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < column; j++)
		{
			
			out << newdata[i][j];
			if (j != column - 1)
				out << ",";
		}
		out << endl;
	}
	out.close();
	out2.close();
    myhand.~hand();
	
	mythread.join();
	return 0;
}
