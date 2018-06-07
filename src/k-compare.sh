python tf_restore_axial_res.py --data=../data/360x240x240-simulation.tif --num_epoch=40 --k_factor=3 --log_suffix="-k-3-patch-60" &&
python tf_restore_axial_res.py --data=../data/360x240x240-simulation.tif --num_epoch=35 --k_factor=4 --log_suffix="-k-4-patch-60" &&
python tf_restore_axial_res.py --data=../data/360x240x240-simulation.tif --num_epoch=30 --k_factor=5 --log_suffix="-k-5-patch-60" &&
python tf_restore_axial_res.py --data=../data/360x240x240-simulation.tif --num_epoch=25 --k_factor=6 --log_suffix="-k-6-patch-60"
