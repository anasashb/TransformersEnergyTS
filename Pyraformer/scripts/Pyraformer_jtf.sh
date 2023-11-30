
# Wind
python3 long_range_main.py -data wind -data_path DEWindh_small.csv -root_path data/  -input_size 168 -predict_step 24 -n_head 6
python3 long_range_main.py -data wind -data_path DEWindh_small.csv -root_path data/  -input_size 168 -predict_step 168 -n_head 6
python3 long_range_main.py -data wind -data_path DEWindh_small.csv -root_path data/  -input_size 168 -predict_step 720 -n_head 6
# synthetic
python3 long_range_main.py -root_path data/ -data synthetic -data_path synthetic.npy -lr 0.001 -window_size [12,7,2] -d_model 256 -d_bottleneck 64 -d_k 64 -d_v 64 -inverse -dropout 0.2 -input_size 168 -predict_step 24;
python3 long_range_main.py -root_path data/ -data synthetic -data_path synthetic.npy -lr 0.001 -window_size [12,7,2] -d_model 256 -d_bottleneck 64 -d_k 64 -d_v 64 -inverse -dropout 0.2 -input_size 168 -predict_step 168;
python3 long_range_main.py -root_path data/ -data synthetic -data_path synthetic.npy -lr 0.001 -window_size [12,7,2] -d_model 256 -d_bottleneck 64 -d_k 64 -d_v 64 -inverse -dropout 0.2 -input_size 168 -predict_step 720;

# Wind
python3 long_range_main.py -data wind -data_path DEWindh_small.csv -root_path data/  -input_size 168 -predict_step 24 -n_head 6 -eval
python3 long_range_main.py -data wind -data_path DEWindh_small.csv -root_path data/  -input_size 168 -predict_step 168 -n_head 6 -eval
python3 long_range_main.py -data wind -data_path DEWindh_small.csv -root_path data/  -input_size 168 -predict_step 720 -n_head 6 -eval
# synthetic
python3 long_range_main.py -root_path data/ -data synthetic -data_path synthetic.npy -lr 0.001 -window_size [12,7,2] -d_model 256 -d_bottleneck 64 -d_k 64 -d_v 64 -inverse -dropout 0.2 -input_size 168 -eval -predict_step 24;
python3 long_range_main.py -root_path data/ -data synthetic -data_path synthetic.npy -lr 0.001 -window_size [12,7,2] -d_model 256 -d_bottleneck 64 -d_k 64 -d_v 64 -inverse -dropout 0.2 -input_size 168 -eval -predict_step 168;
python3 long_range_main.py -root_path data/ -data synthetic -data_path synthetic.npy -lr 0.001 -window_size [12,7,2] -d_model 256 -d_bottleneck 64 -d_k 64 -d_v 64 -inverse -dropout 0.2 -input_size 168 -eval -predict_step 720;