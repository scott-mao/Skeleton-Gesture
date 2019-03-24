#ifndef _ASSIST_HPP_
#define _ASSIST_HPP_
#include <conio.h>
#include <iostream>
#include <fstream>

using namespace std;
// Control the recoding process
void Record(bool *const, string, const int* const, int*const);
//Only for recoding start frame
void Record2(bool *const, string, const int* const, int*const);
//Only for recoding end frame
void Record3(bool *const, string, const int* const);

#endif
