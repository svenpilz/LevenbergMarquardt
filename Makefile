CPPFLAGS=-W -Wall -Wextra -pedantic -std=c++11 
LIBS=`pkg-config --cflags eigen3`

lmtest: lmtest.cpp LevenbergMarquardt.h
	g++ $(CPPFLAGS) $(LIBS) -o lmtest lmtest.cpp
