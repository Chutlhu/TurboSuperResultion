
data_dir=/home/dicarlo_d/Documents/Code/TurboSuperResultion/.cache/Turb2D.hdf5

do_rff=true
do_dvf=true

rff_num=256
python main.py  --dataset Turb2D --data_dir $data_dir   \
                --num_workers 2                         \
                --do_rff $do_rff --rff_num $rff_num     \
                --do_divfree $do_dvf                    \
                --lam_reg 1e-3                          \
                --gpu 1 --max_epoch 5000                \
                --log_every_n_steps 1 --check_val_every_n_epoch 200

# rff_num=1024
# python main.py  --dataset Turb2D --data_dir $data_dir   \
#                 --do_rff $do_rff --rff_num $rff_num     \
#                 --do_divfree $do_dvf                    \
#                 --gpu 1 --max_epoch 5000                \
#                 --log_every_n_steps 1 --check_val_every_n_epoch 200