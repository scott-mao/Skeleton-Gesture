#include "assist.h"

void Record(bool *const stop,string location,const int*const nowframe,int*const labelnum)
{
	char a;
	cout << "Press 'a' to stop the recording" << endl;
	cout << "Press others to record the frame" << endl;
	string K = location + "\\section.csv";
	ofstream out(K);
	int state = 0;//0 for start  1 for end 
	while (true)
	{   
		
		a = _getch();

		if (a == 'a' || *stop == true )
		{
			*stop = true;
			out.close();
			break;
		}
		else
		{   
			if (state == 0)
			{
				(*labelnum)++;
				cout << *labelnum << endl;
				cout << "captured start:" << *nowframe << endl;
				out << *nowframe<<",";
				state = 1;
			}

			else if (state == 1)
			{
				cout << "captured end:" << *nowframe << endl;
				out << *nowframe << endl;
				state = 0;
			}
		}
			

	}

	
}

//Only for recoding start frame
void Record2(bool *const stop, string location, const int*const nowframe, int*const labelnum)
{ 
	char a;
	cout << "Press 'a' to stop the recording" << endl;
	cout << "Press others to record the frame" << endl;
	string K = location + "\\section.csv";
	ofstream out(K);
	
	while (true)
	{

		a = _getch();

		if (a == 'a' || *stop == true)
		{
			*stop = true;
			out.close();
			break;
		}
		else
		{
				(*labelnum)++;
				cout << *labelnum << endl;
				cout << "captured start:" << *nowframe << endl;
				out << *nowframe << endl;
				
			
		}


	}


}

//Only for recoding end frame
void Record3(bool *const stop, string location, const int*const nowframe)
{
	char a;
	string K = location + "\\section_end.csv";
	ofstream out(K);
	while (true)
	{

		a = _getch();

		if (a == 'a' || *stop == true)
		{
			*stop = true;
			out.close();
			break;
		}
		else
		{
			
			
			cout << "captured end:" << *nowframe << endl;
			out << *nowframe << endl;


		}


	}

}

