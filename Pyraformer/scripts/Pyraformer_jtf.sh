
# Wind
python long_range_main.py -root_path data/ -data_path LD2011_2014.txt -data elect \
-input_size 168 -predict_step 24 -n_head 6 -lr 0.00001 -d_model 256;
python long_range_main.py -root_path data/ -data_path LD2011_2014.txt -data elect \
-input_size 168 -predict_step 168 -n_head 6 -lr 0.00001 -d_model 256;
python long_range_main.py -root_path data/ -data_path LD2011_2014.txt -data elect \
-input_size 168 -predict_step 720 -n_head 6 -lr 0.00001 -d_model 256;
# synthetic
python long_range_main.py -root_path data/ -data synthetic -data_path synthetic.npy -lr 0.001 -window_size [12,7,4] -d_model 256 -d_bottleneck 64 -d_k 64 -d_v 64 -inverse -dropout 0.2 -input_size 24 -predict_step 24;
python long_range_main.py -root_path data/ -data synthetic -data_path synthetic.npy -lr 0.001 -window_size [12,7,4] -d_model 256 -d_bottleneck 64 -d_k 64 -d_v 64 -inverse -dropout 0.2 -input_size 168 -predict_step 168;
python long_range_main.py -root_path data/ -data synthetic -data_path synthetic.npy -lr 0.001 -window_size [12,7,4] -d_model 256 -d_bottleneck 64 -d_k 64 -d_v 64 -inverse -dropout 0.2 -input_size 720 -predict_step 720;