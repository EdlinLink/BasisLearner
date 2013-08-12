#include <iostream>
#include <string>
#include <cstring>
#include <fstream>
#include <set>
#include <cmath>
#include <algorithm>
#include <ctime>
#include <stdio.h>

#include "Widths.h"
#include "Posi.h"
#include "MyMath.h"

#define ARMA_NO_DEBUG
#include "armadillo"

#define X_row 16839
#define X_col 780
#define Y_row 16839


//Trainend >= Node
#define Node 50
#define Layer 7
#define Trainend 10839
#define LambdaRange 1e-6
#define LambdaCount 5

#define INF 999999999
#define INF_N -9999999999
using namespace std;
using namespace arma;


long edlin = 0;

mat X_1;
mat Y;


Widths widths;
long trainend;
string BuildMethodFirstLayer;
int batchSize = Node;
double tol;


mat Ytrain;
set<int> classLabels;
mat F, OF;
long F_row, F_col;
long OF_row, OF_col;


mat lambdaRange;
mat Result_train;
mat Result_test;


clock_t start, finish;


bool ProgramInit(){
	start = clock();
	srand((unsigned)time(NULL));
	if(Trainend < Node){
		cerr << "Not satisfy: Trainend >= Node"<<endl;
		return false;
	}
	return true;
}


void LoadData(){
	FILE *fp;
	
	Mat<int> ran = randperm(X_row);
	long select_row;
	
	fp = fopen("X.txt", "r");
	X_1.ones(X_row, 1+X_col);
	for(long count=0; count<X_row; count++){
		select_row = ran(0, count);
		select_row = count;
		for(long col=1; col<X_col+1; col++){
			fscanf(fp, "%lf", &X_1(select_row,col));
		}
	}
	fclose(fp);

	fp = fopen("YY.txt", "r");
	Y.zeros(Y_row, 1);
	for(long count=0; count<Y_row; count++){
		select_row = ran(0, count);
		select_row = count;
		fscanf(fp, "%lf", &Y(select_row,0));
	}
	fclose(fp);
}



void SetParameter(){
	widths.node.clear();
	for(int layer=0; layer<Layer; layer++){
		widths.node.push_back(Node);
	}
	
	trainend = Trainend;
	BuildMethodFirstLayer = "exact";
	//batchSize = 100;
	tol = 1e-9;	
}



void PreProcessData(){
	cout << "Building F for widths: [";
	for(int i=0; i<widths.node.size(); i++){
		cout<<widths.node.at(i)<<" ";
	}
	cout << "]" << endl;
	
	Ytrain = Y.submat(0,0,trainend-1,0);
	
	classLabels.clear();
	int count=0;
	for(int i=0; i<trainend; i++){
		classLabels.insert(Ytrain.at(i,0));
	}
	
	F_row = X_row;
	F_col = widths.sum();
	F.zeros(F_row, F_col);
	
	OF_row = trainend;
	OF_col = widths.sum();
	OF.zeros(OF_row, OF_col);
}




void CreateInputLayer(){
	clock_t a = clock();
	/*
	mat X_1;
	X_1.ones(X_row, 1+X_col);
	for(int i=0; i<X_row; i++){
		for(int j=1; j<1+X_col; j++){
			X_1(i, j) = X.at(i,j-1);
		}
	}*/
	
	mat W;
	if(BuildMethodFirstLayer == "exact"){
		W = orthonormalize(X_1.submat(0,0,trainend-1,X_col), trainend, X_col+1, widths.node.front());
	}
	else if(BuildMethodFirstLayer == "approx"){
		cerr << "Not implement yet!\n";
	}
	else{
		cerr << "Unknown BuildMethodFirstLayer" << endl;
	}
	
	mat f = X_1 * W;
	
	long f_col = widths.node.front();
	
	for(int i=0; i<F_row; i++){
		for(int j=0; j<f_col; j++){
			F(i,j) = f(i,j);
		}
	}
	
	for(int i=0; i<trainend; i++){
		for(int j=0; j<f_col; j++){
			OF(i,j) = F.at(i,j);
		}
	}	
	
	//normalize
	for(int i=0; i<f_col; i++){
		double sum = 0;
		for(int j=0; j<trainend; j++){
			sum += (F.at(j,i) * F.at(j,i));
		}
		double sq = sqrt(sum/trainend);
		for(int j=0; j<F_row; j++){
			F(j,i) = F.at(j,i) / sq;
		}
	}
	
	clock_t b = clock();
	cout <<"### IL TIME = "<<(b-a)/CLOCKS_PER_SEC<<endl;
}



void CreateIntermediateLayer(){
	clock_t a = clock();
	for(int t=2; t<=Layer; t++){
		long beginThis = widths.sum(1,t-1) + 1;
		long beginLast = widths.sum(1,t-2) + 1;
		
		mat V;
		long V_row = trainend;
		long V_col = 1;							//initial for 2-classify
		if(classLabels.size()>2){
			V.zeros(trainend, classLabels.size());
			V_col = classLabels.size();
			for(int i=0; i<classLabels.size(); i++){
				for(int j=0; j<trainend; j++){
					if(Ytrain.at(j,0)==i)
						V(j,i) = 1;
				}
			}
		}
		else{
			V.ones(trainend, 1);								//Initial V to all 1, meaning CLASS 1
			set<int>::iterator it;						
			it=classLabels.begin();
			for(int i=0; i<trainend; i++){					//if the Ytrain is not same as the first label, set -1, meaning CLASS 2 
				if(double(Ytrain.at(i,0)==(*it)) < 0.1)
					V(i,0) = -1;
			}
		}
		
		mat of = OF.submat(0 ,0 ,OF_row -1, beginThis-1 -1);
		mat of_2 = of*(of.t()*V);
		for(int i=0; i<V_row; i++){
			for(int j=0; j<V_col; j++){
				V(i, j) = V.at(i, j) - of_2(i, j);
			}
		}
		
		int r = beginThis;
		while(r <= beginThis+widths.node.at(t-1)-1){
			mat scores;
			scores.zeros(widths.node.front(), widths.node.at(t-1-1));
			
			mat OV = orth(V, V_row, V_col);
			for(int i=0; i<widths.node.front(); i++){ //Compute scores
				mat Ci;
				long Ci_col = widths.node.at(t-1-1);
				long Ci_row = trainend;
				Ci.zeros(trainend, Ci_col);
				
				for(int b=0; b<Ci_col; b++){
					for(int a=0; a<trainend; a++){
						Ci(a, b) = F.at(a,i)*F.at(a,beginLast+b-1);
					}
				}
				
				of = OF.submat(0, 0, OF_row-1, r-1-1);
				of_2 = of * (of.t()*Ci);
				
				for(int a=0; a<Ci_row; a++){
					for(int b=0; b<Ci_col; b++){
						Ci(a, b) = Ci.at(a, b) - of_2.at(a, b);
					}
				}
				
				
				for(int a=0; a<widths.node.at(t-2); a++){
					double normCi = 0;
					for(int b=0; b<Ci_row; b++){
						normCi += (Ci.at(b,a)*Ci.at(b,a));
					}
					normCi = sqrt(normCi);
					for(int b=0; b<Ci_row; b++){
						Ci(b,a) = Ci(b,a)/normCi;
					}
				}
				
				
				mat OV_T__Ci = OV.t() * Ci;
				
				for(int a=0; a<widths.node.at(t-2); a++){
					scores(i, a) = (OV_T__Ci.at(0,a) * OV_T__Ci.at(0,a));
					if(scores.at(i,a)<tol)
						scores(i, a) = INF_N;
				}
			}
			
			Posi *M = new Posi[scores.size()];
			for(int j=0; j<widths.node.at(t-2); j++){//col
				for(int i=0; i<widths.node.front(); i++){//row
					M[i*widths.node.at(t-2) + j].value = scores.at(i, j);
					M[i*widths.node.at(t-2) + j].x = i;
					M[i*widths.node.at(t-2) + j].y = j;
				}
			}
			stable_sort(M, M+scores.size(), cmp);
			
			long numNewColumns = beginThis+widths.node.at(t-1)-r;
			if(numNewColumns > batchSize)
				numNewColumns = batchSize;
			
			
			mat Cchosen;
			Cchosen.zeros(trainend, numNewColumns);
			mat OC;
			OC.zeros(trainend, numNewColumns);
			long l = 1;
			long ind = 0;
			
			double normOCl;
			
			while(l<=numNewColumns){
				
				mat f = F;
				
				clock_t x = clock();
				for(int i=0; i<F_row; i++){
					F(i,r-1+l-1) = f(i,M[ind].x) * f(i,beginLast+M[ind].y-1);
				}
				clock_t y = clock();
				
				for(int i=0; i<trainend; i++){
					Cchosen(i, l-1) = F(i, r-1+l-1);
				}
				
				mat subOF = OF.submat(0, 0, OF_row-1, r-1-1);				
				mat subOF_2 = of*(subOF.t()*Cchosen.col(l-1));
				
				for(int i=0; i<trainend; i++){
					OC(i,l-1) = Cchosen(i, l-1) - subOF_2(i, 0);
				}
				
				if(l!=1){
					mat	subOC = OC.submat(0, 0, trainend-1, l-2);
					mat subOC_2 = subOC * (subOC.t()*OC.col(l-1)); 
					for(int i=0; i<trainend; i++){
						OC(i,l-1) = OC(i,l-1) - subOC_2(i,0);
					}
				}
				
				double sumOC = 0;
				for(int i=0; i<trainend; i++){
					sumOC+=(OC(i,l-1) * OC(i,l-1));
				}
				normOCl = sqrt(sumOC);
				
				if(normOCl > tol){			//Accept new vector if it's linearly independent from previous ones
					for(int i=0; i<trainend; i++){
						OC(i, l-1) = OC(i, l-1)/normOCl;
					}
					
					double sumup = 0;
					for(int i=0; i<trainend; i++){
						sumup+=(F(i,r-1+l-1) * F(i,r-1+l-1));
					}
					double sq = sqrt(sumup/trainend);
					
					for(int i=0; i<F_row; i++){
						F(i, r-1+l-1) = F(i, r-1+l-1)/sq;
					}
					l++;
				}

				ind++;
				
				if(ind >= scores.size()){
					if(l==1)
						cerr << "No linearly-independent candidates were found in mini-batch"<<endl;
					else
						cerr<<"Warning: Only " << l-1 <<"(out of mini-batch size "<<batchSize<<") linearly-independent candidate vectors were found and added."<<endl;
					break;	
				}
			}
			
			if(normOCl > tol)
				l--;
			
			for(int i=0; i<OF_row; i++){
				for(int j=r; j<=r-1+l; j++){
					OF(i, j-1) = OC(i, j-r);
				}
			}
			
			mat subOC = OC.submat(0, 0, trainend-1, l-1);
			mat tmp = subOC * subOC.t() * V;
			
			for(int i=0; i<V_row; i++){
				for(int j=0; j<V_col; j++){
					V(i,j) = V(i,j) - tmp(i,j);
				}			
			}
			
			r+=l;
			
			long Min = widths.node.at(t-1);
			if(Min>r-beginThis)
				Min = r-beginThis;
			
			cout <<"Built "<< Min <<" out of "<< widths.node.at(t-1)<<" elements in layer "<< t <<endl;
			
		}
	}
	
	clock_t b = clock();
	cout <<"### ML TIME = "<<(b-a)/CLOCKS_PER_SEC<<endl;
}



void CreateOutputLayer(){
	clock_t a = clock();
	lambdaRange.zeros(1,LambdaCount);
	lambdaRange(0,0) = LambdaRange;
	for(int i=1; i<LambdaCount; i++){
		lambdaRange(0,i) = lambdaRange(0,i-1)*10;
	}
	
	
	Result_train.zeros(widths.node.size(),LambdaCount);
	Result_test.zeros(widths.node.size(), LambdaCount);
	
	cout << "Building Output Layer\n trainend = "<<trainend<<". widths = [";
	for(int i=0; i<widths.node.size(); i++)
		cout<<widths.node.at(i)<<" ";
	cout<<"], lambdaRange = [";
	for(int i=0; i<LambdaCount; i++)
		cout<<LambdaRange*pow(10.0,i)<<" ";
	cout<<"]"<<endl<<endl;
	
	
	string lossType;
	if(classLabels.size()>2)
		lossType = "multiclass_hinge";
	else
		lossType = "hinge";
	
	int szCounter = 0;
	vector<int> cumsum_widths;
	cumsum_widths.clear();
	cumsum_widths.push_back(widths.node.front());
	for(int i=1; i<widths.node.size(); i++){
		int tmp = cumsum_widths.back();
		cumsum_widths.push_back(tmp+widths.node.at(i));
	}
	
	mat preds;
	preds.zeros(X_row,1);
	for(int i=0; i<cumsum_widths.size(); i++){
		long sz = cumsum_widths.at(i);
		long lambdaCounter = 0;
		for(int j=0; j<LambdaCount; j++){
			double lambda = lambdaRange(0,j);
			
			cout << "Training depth "<<szCounter+2<<", lambda "<<lambda<<endl;

			if(lossType == "hinge"){
				mat subF = F.submat(0, 0, trainend-1, sz-1);
				mat subY = Y.submat(0, 0, trainend-1, 0);
				mat w = SGD_hinge(subF, trainend, sz, subY, lambda);
				
				preds = F.submat(0, 0, F_row-1, sz-1)*w;
				for(int a=0; a<X_row; a++){
					if(preds(a,0)>0)
						preds(a,0) = 1;
					else
						preds(a,0) = -1;
				}
			}
			else{
				long minY = INF;
			
				set<int>::iterator it;
				for(	it=classLabels.begin(); it!=classLabels.end(); it++){
					if(minY>*it)
						minY = *it;
				}
				
				for(int i=0; i<Y_row; i++){
					Y(i,0) = Y(i,0) - minY + 1;
				}
				
				mat subF = F.submat(0, 0, trainend-1, sz-1);
				mat subY = Y.submat(0, 0, trainend-1, 0);
				
				mat w = MC_SGD(subF, trainend, sz, subY, classLabels.size(), lambda);

				subF = F.submat(0, 0, F_row-1, sz-1);
				mat temp = subF*w;
				for(int a=0; a<F_row-1; a++){
					long Max = INF_N;
					int pre;
					for(int b=0; b<classLabels.size();b++){
						if(Max<temp(a,b)){
							Max = temp(a,b);
							pre = b;
						}
					}
					preds(a,0) = pre+1;
				}
				
				for(int a=0; a<F_row-1; a++){
					preds(a,0) = preds(a,0)+minY-1;
				}
			}
		
			long error = 0;
			for(int a=0; a<trainend; a++){
				if(preds(a,0)!=Y(a,0))
					error++;
			}
			Result_train(szCounter, lambdaCounter) = error/double(trainend);
			
			error = 0;
			for(int a=trainend; a<Y_row; a++){
				if(preds(a,0)!=Y(a,0))
					error++;
			}
			Result_test(szCounter, lambdaCounter) = error/double(Y_row-trainend);
			
			cout <<"Train Error: "<<Result_train(szCounter, lambdaCounter)<<", Test Error: "<<Result_test(szCounter, lambdaCounter)<<endl<<endl;
			lambdaCounter++;
			
		}
		
		szCounter++;
	}
	clock_t b = clock();
	cout <<"### OL TIME = "<<(b-a)/CLOCKS_PER_SEC<<endl;
}



void BestResult(){
	double bestTestError = 1.00;
	long best_i,best_j;
	
	for(int i=0; i<widths.node.size(); i++){
		for(int j=0; j<LambdaCount; j++){
			if(bestTestError > Result_test(i,j)){
				
				bestTestError = Result_test(i,j);
				best_i = i;
				best_j = j;
			}
		}
	}
	Result_train.zeros(widths.node.size(),LambdaCount);
	
	cout << "Best Test Error Result: " <<bestTestError*100<<"%"<<endl;
	cout << " - Architecture [";
	for(int i=0; i<=best_i; i++){
		cout << widths.node.at(i) <<" ";
	}
	
	cout<<"]\n - lambda "<<lambdaRange(0,best_j)<<"(no. "<<best_j+1<<" in lambdaRange)"<<endl;
}



void ProgramEnd(){
	finish=clock();
    cout<<"TIME = "<<(finish-start)/CLOCKS_PER_SEC<<endl;
}




int main(){

	if(!ProgramInit())
		return 1;
	
	LoadData();
	SetParameter();
	PreProcessData();		
	
	CreateInputLayer();
	CreateIntermediateLayer();
	CreateOutputLayer();
	
	BestResult();
	
	ProgramEnd();
	return 0;
}

