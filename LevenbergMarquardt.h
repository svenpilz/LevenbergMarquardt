#ifndef LEVENBERGMARQUARDT_H
#define	LEVENBERGMARQUARDT_H

/*
 * Based on the paper “A Brief Description of the Levenberg-Marquardt Algorithm Implemented by levmar” by Manolis I. A. Lourakis.
 */

#include <stdexcept>
#include <limits>
#include <functional>
#include <Eigen/Dense>

class LevenbergMarquardt {
public:
	static constexpr double EPSILON = 10e-8;

	LevenbergMarquardt() :
		tau(10e-3), e1(10e-15), e2(10e-15), e3(10e-15),
		damping(2.0), maxIterations(100) { }
	
	Eigen::VectorXd operator()(Eigen::VectorXd initialGuess, Eigen::VectorXd x, std::function<Eigen::VectorXd(Eigen::VectorXd)> func) {
		double v = damping; // Damping.
		auto p = initialGuess;
		auto J = LevenbergMarquardt::jacobi(func, p, initialGuess.rows(), x.rows());
		Eigen::MatrixXd A = J.transpose() * J;
		Eigen::VectorXd error = x - func(p);
		Eigen::MatrixXd g = J.transpose() * error;
		bool stop = g.cwiseAbs().maxCoeff() <= e1;
		double u = tau * A.diagonal().maxCoeff();
		const auto I = Eigen::MatrixXd::Identity(A.rows(), A.cols());

		for (int k = 0; k < maxIterations && !stop; k++) {
			double rho = 0;
			
			do {
				Eigen::VectorXd delta = (A + u*I).colPivHouseholderQr().solve(g);
				
				if (delta.norm() <= e2*p.norm()) {
					stop = true;
					
				} else {
					Eigen::VectorXd newGuess = p + delta;
					Eigen::VectorXd newValue = func(newGuess);
					rho = error.norm()*error.norm() - (x - newValue).norm()*(x - newValue).norm();
					rho /= (delta.transpose() * (u*delta+g))[0];
					
					if (rho > 0) {
						p = newGuess;
						J = LevenbergMarquardt::jacobi(func, p, p.rows(), x.rows());
						A = J.transpose() * J;
						error = x - newValue;
						g = J.transpose() * error;
						stop = g.cwiseAbs().maxCoeff() <= e1 || error.norm()*error.norm() <= e3; // max(g) <= e1 OR length(error)^2 <= e3
						
						u = u * std::max(1.0/3.0, 1.0 - (2.0*rho-1.0)*(2.0*rho-1.0)*(2.0*rho-1.0)); // u*max(1/3, (2*rho-1)^3)
						v = damping;
					} else {
						u *= v;
						v *= damping;
					}
				}	
			} while(rho > 0 && !stop && !std::isnan(u) && !std::isinf(u));
		}
		
		if (std::isnan(u) || std::isinf(u)) {
			throw std::domain_error("LevenbergMarquardt: µ is NAN or INF.");
		}
		
		return p;
	}
	
	static Eigen::MatrixXd jacobi(std::function<Eigen::VectorXd(Eigen::VectorXd)> func, Eigen::VectorXd p, int variables, int dimensions) {
		Eigen::MatrixXd m(dimensions, variables);
		
		for (int v = 0; v < variables; v++) {
			double value = p[v];
			
			p[v] = value - EPSILON;
			Eigen::VectorXd l = func(p);
			
			p[v] = value + EPSILON;
			Eigen::VectorXd r = func(p);
			
			p[v] = value;
			
			for (int d = 0; d < dimensions; d++) {
				m(d, v) = (r[d] - l[d]) / (2 * EPSILON);
			}
			
		}
		
		return m;
	}
	
private:
	double tau;
	double e1;
	double e2;
	double e3;	
	double damping;
	int maxIterations;
};

#endif	/* LEVENBERGMARQUARDT_H */
