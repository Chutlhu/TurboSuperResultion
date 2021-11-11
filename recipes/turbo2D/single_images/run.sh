
data_dir=/home/dicarlo_d/Documents/Code/TurboSuperResultion/.cache/Turb2D.hdf5

mkdir -p figures

do_rff=true
do_dvf=true
lam_div=0
lam_pde=0
lam_reg=0
lam_sfn=0
lam_spec=0

rff_num=1024

python main.py  --dataset Turb2D --data_dir $data_dir   \
                --num_workers 1                         \
                --do_rff $do_rff --rff_num $rff_num     \
                --do_divfree $do_dvf                    \
                --lam_sfn $lam_sfn                      \
                --sfn_num_centers 50                    \
                --sfn_num_increments 3                  \
                --sfn_patch_dim   30                    \
                --lam_spec $lam_spec                    \
                --lam_pde  $lam_pde                     \
                --lam_div  $lam_div                     \
                --lam_reg  $lam_reg                     \
                --gpus 1 --max_epoch 5000                \
                --log_every_n_steps 1 --check_val_every_n_epoch 200

# rff_num=1024
# python main.py  --dataset Turb2D --data_dir $data_dir   \
#                 --do_rff $do_rff --rff_num $rff_num     \
#                 --do_divfree $do_dvf                    \
#                 --gpu 1 --max_epoch 5000                \
#                 --log_every_n_steps 1 --check_val_every_n_epoch 200