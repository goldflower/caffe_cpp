cp convert_pickup_seq.cpp ../tools/
cd ../build
make all && make install
cd -
../build/tools/convert_pickup_seq  database lmdb_file_list.txt lmdb_database
