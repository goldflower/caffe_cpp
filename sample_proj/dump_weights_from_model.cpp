#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <ostream>
#include "boost/scoped_ptr.hpp"
#include "gflags/gflags.h"
#include "glog/logging.h"

#include <caffe/caffe.hpp>
using namespace caffe;
using namespace std;


int main(int argc, char** argv){
  Caffe::set_mode(Caffe::CPU);
  Net<float> fc(argv[1], caffe::TEST);
  fc.CopyTrainedLayersFrom(argv[2]);
  vector<Blob<float>*> blobs = fc.learnable_params();
  string weight_file_name = argv[3];
  ofstream f(weight_file_name);
  f << "axis='xyzxgygzg'\n2\n84 120 1 84\n";
  for (int i = 0; i < blobs.size(); i++){
    string shape_info = blobs[i]->shape_string();
    vector<int> shape_info_vec = blobs[i]->shape();
    if (shape_info_vec.size() == 1){
      shape_info_vec.push_back(1);
    }
    float* data = blobs[i]->mutable_cpu_data();
    for (int j = 0; j < shape_info_vec[0]; j ++){
	for (int k = 0; k < shape_info_vec[1]; k++){
           f << data[j * shape_info_vec[1] + k] << " ";
           //cout << data[j * shape_info_vec[1] + k] << " ";
	}
	f << "\n";
	
    }
    
//    cout << data[0] << endl;
//    cout << shape_info << endl;
  }
  /*
  cout << blobs.size() << " " << endl;
  string shape_info = blobs[0]->shape_string();
  cout << shape_info << endl;
  
  for (int i = 0; i < blobs.size(); i++){
    Blob<type>* blob = blobs[i];
    for (int j = 0; j < blobs[i].size(); j++){
      cout << blobs[i][j] << " ";
    }
    cout << endl;
  }
  */
  return 0;
}
