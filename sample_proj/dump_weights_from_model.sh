cp dump_weights_from_model.cpp ../tools/
cd ../build
cmake ..
make all && make install
cd -
../build/tools/dump_weights_from_model train_val.prototxt checkpoints/ckpt_iter_00005000_loss_inf_.caffemodel abc.txt

