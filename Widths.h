#pragma once
#include <vector>
using namespace std;


class Widths{
public:
	vector<int> node;

	long sum(){
		long total=0;
		for(int i=0; i<node.size(); i++){
			total+=node.at(i);
		}
		return total;
	}

	long sum(long a, long b){
		long total = 0;
		for(int i=a; i<=b; i++){
			total+=node.at(i);
		}
		return total;
	}


};


