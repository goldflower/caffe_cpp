#include <algorithm>
#include <fstream>

#include <string>
#include <utility>
#include <vector>
#include <iostream>
#include <sstream>
#include <iterator>
#include <numeric>

#include "boost/scoped_ptr.hpp"
#include "gflags/gflags.h"
#include "glog/logging.h"

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/format.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"

using namespace std;
using namespace caffe;
using std::pair;
using boost::scoped_ptr;

DEFINE_string(backend, "lmdb",
		"The backend {lmdb, leveldb} for storing the result");

void display_progress(float progress){
  int barWidth = 70;
  cout << "[";
  int pos = barWidth * progress;
  for (int i = 0; i < barWidth; i++){
    if (i < pos) cout << "=";
    else if (i == pos) cout << ">";
    else cout << " ";
  }
  cout << "] " << int(progress * 100.0) << " %\r";
  cout.flush();
}

float get_min(vector<float> vec, int start, int end){
  float min = INT_MAX;
  for (int i = start; i < end; i++){
    if (vec[i] < min) min = vec[i];
  }
  return min;
}

float get_max(vector<float> vec, int start, int end){
  float max = -INT_MIN;
  for (int i = start; i < end; i++){
    if (vec[i] > max) max = vec[i];
  }
  return max;
}

bool heuristic_check(vector<float> seq)
{
    int y_th = 5;
    int z_th = 5;
    // put down check
    if (seq[axis_len] - seq[axis_len * 2 - 1] < -3)
        return true;
    // head down check
    for (int i = 0; i < 3; i++)
    {
        if (seq[axis_len + i] < 0)
        {
            return true;
        }
    }
    // steady check
    float y_M = get_max(seq, axis_len, 2 * axis_len - 1);
    float y_m = get_min(seq, axis_len, 2 * axis_len - 1);
    float z_M = get_max(seq, 2 * axis_len, 3 * axis_len - 1);
    float z_m = get_min(seq, 2 * axis_len, 3 * axis_len - 1);
    if (((y_M - y_m) < y_th) && ((z_M - z_m) < z_th))
    {
        return true;
    }
    // cover check
    for (int i = 0; i < 3; i++)
    {
        if (seq[axis_len * 2 + i] < 0)
        {
            return true;
        }
    }
    return false;
}

vector<float> getSequence(const string &filename){
    ifstream infile(filename);
    string line;
    vector<float> vec;
    vector<float> result;
    while (getline(infile, line))
    {
        stringstream s(line);
        float num;
        while (s >> num)
        {
            vec.push_back(num);
            if (s.peek() == ',')
            {
                s.ignore();
            }
        }
    }
    for (int i = 0; i < 6; i++)
    { // for NN input
        for (int j = i; j < vec.size(); j += 6)
        {
            result.push_back(vec[j]);
        }
    }
    return result;    
}


// just convert 1-d sequence to datum, not for general perpose
bool ReadSensorsToDatum(const string &filename, int label,
		        bool do_check, Datum* datum)
{
    //cout << filename << endl;
    vector<float> vec = getSequence(filename);
    if (vec.size() != axis_len * 6){
     // cout << vec.size() << endl;
      cout << "cond1" << endl;
      return false;
    }
    if (do_check && label == 1 && heuristic_check(vec)){
    //  cout << "cond2" << endl;
      return false;
    }
    datum->set_channels(1);
    datum->set_height(1);
    datum->set_width(axis_len * 6);
    google::protobuf::RepeatedField<float>* datumFloatData = datum->mutable_float_data();
    for (int i = 0; i < vec.size(); i++){
   //   cout << vec[i] << " ";
      datumFloatData->Add(vec[i]);
    }
    datum->set_label(label);
    return true;
}


int main(int argc, char** argv){

#ifdef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif
  const bool check_size = FLAGS_check_size;
  gflags::SetUsageMessage("Convert a set of pickup sequences to the leveldb/lmdb\n"
		  "format uses as input for Caffe.\n"
		  "convert_pickup_seq [FLAGS] ROOTFOLDER LISTFILE DB_NAME");
  // reading data / label pairs,
  // => [(filename1, label1), (...), ...]
  // according to this pairs we can get the information of each data
  ifstream infile(argv[2]);
  cout << "file listed in: " << argv[2] << endl;
  vector<std::pair<std::string, int> > lines;
  string line;
  size_t pos;
  int label;
  if (argc < 4){
    gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/convert_pickup_seq");
  }
  while (getline(infile, line)) {
    pos = line.find_last_of(' ');
    label = atoi(line.substr(pos + 1).c_str());
    lines.push_back(make_pair(line.substr(0, pos), label));
  }
  if (FLAGS_shuffle) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    shuffle(lines.begin(), lines.end());
  }
  LOG(INFO) << "A total of " << lines.size() << " files.";
  
  // create new DB
  scoped_ptr<db::DB> db(db::GetDB(FLAGS_backend));
  db->Open(argv[3], db::NEW);
  cout << "db name: " << argv[3] << endl;
  scoped_ptr<db::Transaction> txn(db->NewTransaction());

  // storing to db
  string root_folder(argv[1]);
  cout << "root folder: " << root_folder << endl;
  int count = 0;
  int data_size = 0;
  bool data_size_initialized = false;
  int once = 0;
  for (int line_id = 0; line_id < lines.size(); ++line_id){
    if (line_id % 100 == 0){
      display_progress(line_id / (lines.size()+0.0f));
    }
    bool status;
    Datum datum;
    //cout << "Handling " << lines[line_id].first << endl;
    status = ReadSensorsToDatum(root_folder + '/' + lines[line_id].first, lines[line_id].second,
		                true, &datum);
    if (!status) continue;
    // sequential
    string key_str = caffe::format_int(line_id) + "_" + lines[line_id].first;

    // put in db
    string out;
    CHECK(datum.SerializeToString(&out));
    txn->Put(key_str, out);
    /*
    for (int i = 0; i < datum->mutable_float_data().size(); i++){
      cout << datum->mutable_float_data()[i] << " ";
    }
    */
    if (++count % 1000 == 0){
      // commit db
      txn->Commit();
      txn.reset(db->NewTransaction());
      LOG(INFO) << "Processed " << count << " files.";
    }
    //cout << "Success for " << lines[line_id].first << endl;
  }
  // write the last batch
  if (count % 1000 != 0){
    txn->Commit();
    LOG(INFO) << "Processed " << count << " files.";
  }
  return 0;
}
