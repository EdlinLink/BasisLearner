#pragma once
#include <iostream>
#include <string>
#include <ctime>
#include "armadillo"

using namespace std;
using namespace arma;



bool cmp(const Posi& a, const Posi& b){
	return a.value>=b.value;
}



Mat<int> randperm(long m){	
	Mat<int> w;
	w.zeros(1,m);

	for(int i=0; i<m; i++){
		w.at(0,i) = i;
	}
	int r;
	int tmp;
	for(int i=0; i<m; i++){
		r = rand()%m;
		tmp = w.at(0,i);
		w.at(0,i) = w.at(0,r);
		w.at(0,r) = tmp;
	}	

	return w;
}





mat SGD_hinge(mat X, long M, long N, mat Y, double lam){
	
	X = X.t();
	double factor = sqrt(2/lam);
	int num_epochs = 50;
	long m = M;
	long n = N;
	
	mat w;
	w.zeros(1,n);
	Mat<int> inds;
	
	inds = randperm(m);
	double t = 1;
	
	for(long epoch=1; epoch<=num_epochs; epoch++){
		for(long i=0; i<m; i++){
			long ind = inds.at(0,i);
			mat temp = (Y.row(ind)*w*X.col(ind));
			
			if(1 > temp.at(0,0)){
				mat tmp = Y.row(ind)*((X.col(ind)).t());
				
				for (int j=0; j<n; j++) {
					w(0,j) = (1-1/t)*w.at(0,j) + tmp.at(0,j)/(lam*t);
				}
			}
			else{
				w = (1-1/t)*w;
			}
			
			double Min = 0;
			for(int i=0; i<n; i++){
				Min+=(w.at(0,i)*w.at(0,i));
			}
			Min = sqrt(Min);
			if(Min>1)
				Min = 1;
			
			for(int j=0; j<n; j++){
				w(0,j) = Min * w.at(0,j);
			}
			t++;
		}
	}
	
	return w.t();
}




mat MC_SGD(mat X, long M, long N, mat Y, int ClassLabels, double lam){
	X = X.t();
	long k = ClassLabels;
	
	int num_epochs = 100;
	long m = M;
	long n = N;
	
	mat w;
	w.zeros(k,n);
	Mat<int> inds;
	
	
	double t=1;
	for(int epoch=1; epoch<=num_epochs; epoch++){
		inds = randperm(m);
		for(int i=0; i<m; i++){
			long ind = inds.at(0,i);
			mat pred = w*X.col(ind);
			
			double val=-999999;
			long j;

			for(int a=1; a<=k; a++){
				int tmp = 0;
				if(a!=Y.at(ind))
					tmp = 1;
				double temp = tmp + pred.at(a-1,0) - pred.at(Y(ind)-1,0);
				
				if (val < temp) {
					val = temp;
					j = a;
				}
			}
			
			w = (1-1/t)*w;
			if(val>0){
				for(int a=0; a<n; a++){
					w(Y(ind)-1,a) = w.at(Y(ind)-1,a) + X.at(a,ind)/(lam*t);
				}
				for(int a=0; a<n; a++){
					w(j-1,a) = w.at(j-1,a) - X.at(a,ind)/(lam*t);
				}
			}
			t++;
		}
	}

	return w.t();
}




mat orthonormalize(mat G, long row, long col, int width){
	clock_t a = clock();
	
	mat U;
	vec D;
	mat W;
	
	
	svd(U, D, W, G);
	
	vector<int> needIndex;
	needIndex.clear();
	for(int i=0; i<D.size(); i++){
		if(D.at(i)>0.0001){
			needIndex.push_back(i);	//record form 1
		}
	}
	
	long Min = width;
	if(Min > needIndex.size())
		Min = needIndex.size();
	
	long w_row = col;
	mat new_w;
	new_w.zeros(w_row, Min);
	
	for(int i=0; i<Min; i++){
		for(int j=0; j<w_row; j++){
			new_w(j, i) = W.at(j, needIndex.at(i));
		}
	}
		
	mat B = G*new_w;
	mat target_w;
	target_w.zeros(w_row, Min);
	
	for(int i=0; i<Min; i++){
		double sum=0;
		for(int j=0; j<row; j++){
			sum+=(B.at(j, i)*B.at(j, i));
		}
		double sq = sqrt(sum);
		
		for(int j=0; j<w_row; j++){
			target_w(j,i) = new_w.at(j,i)/sq;
			
		}
		
	}
	
	clock_t b = clock();
	cout <<"### OR TIME = "<<(b-a)/CLOCKS_PER_SEC<<endl;
	
	return target_w;
}


mat orth(mat A, long row, long col){
	mat Q;
	vec S;
	mat tmp;
	svd(Q, S, tmp, A);
	long Q_row = row;
	long Max = row;
	long S_edge = S.size();
	if(S_edge>Max){
		Max = S_edge;
		
	}
	
	
	if(!S.is_empty()){
		double eps = 2.2204e-16;
		double tol = Max*S.at(0,0)*eps;
		long r = 0;
		for(int i=0; i<S_edge; i++){
			if(S.at(i,i) > tol)
				r++;
		}
		mat newQ;
		newQ.zeros(Q_row, r);
		for(int i=0; i<Q_row; i++){
			for(int j=0; j<r; j++){
				newQ(i, j) = Q.at(i, j);
			}
		}
		return newQ;
	}
	return Q;
}

