#include "LevenbergMarquardt.h"

#include <iostream>

using namespace std;

Eigen::VectorXd f(Eigen::VectorXd x) {
	Eigen::VectorXd result(1);
	result[0] = x.dot(x);
	
	return result;
}

int main() {
	LevenbergMarquardt optimizer;
	Eigen::VectorXd guess(1);
	Eigen::VectorXd target(1);
	
	guess[0] = 1;
	target[0] = 4;
	
	cout << "guess=" << guess.transpose() << ", target=" << target.transpose() << endl;
	
	auto p = optimizer(guess, target, f);
	
	cout << "f(" << p << ") = " << f(p) << endl; 
	
	return 0;
}
