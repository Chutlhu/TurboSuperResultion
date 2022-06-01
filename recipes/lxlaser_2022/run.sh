source /home/dicarlo_d/Documents/Code/TurboSuperResultion/venv/bin/activate
data_dir=/home/dicarlo_d/Documents/Code/TurboSuperResultion/.cache/Turb2D.hdf5

exp_name='LXLASER2022_RFFMLP_Turb2D'

log_dir=logs_comparison
res_dir=results_comparison
fig_dir=figures

mkdir -p $fig_dir
mkdir -p $log_dir
mkdir -p $res_dir

seed=666

max_epoch=5000

train_ppp=0.03
val_ppp=0.25
test_ppp=1

ni=4


for run in 333
do

for train_ppp in 0.01 0.03 0.10
do

for do_dvf in false
do

for mlp_layers_num in 3
do

for mlp_layers_dim in 256
do

for rff_num in 256
do

for lam_sdiv in 0 1e-5 1e-3
do

for lam_sfn in 0 1e-5 1e-3
do

for lam_curl in 0 1e-5 1e-3
do

name="$exp_name-run:$run-train_ppp:$train_ppp-mlp:$mlp_layers_num-rff_num:$rff_num-df:$do_dvf-lsdv:$lam_sdiv-lgrd:$lam_grads-lcrl:$lam_curl-lsfn:$lam_sfn-ni:$ni"
echo $name

python main.py  --exp_name $name                        \
                --logs_dir $log_dir                     \
                --res_dir $res_dir                      \
                --seed $seed                            \
                --dataset Turb2D --data_dir $data_dir   \
                --time_idx $run                         \
                --train_do_offgrid true                    \
                --val_do_offgrid   true                    \
                --test_do_offgrid  true                    \
                --train_ppp         $train_ppp          \
                --val_ppp           $val_ppp            \
                --test_ppp          $test_ppp           \
                --num_workers 1                         \
                --name  $name                           \
                --mlp_layers_num $mlp_layers_num        \
                --mlp_layers_dim $mlp_layers_dim        \
                --do_rff true                           \
                --rff_num $rff_num                      \
                --do_divfree $do_dvf                    \
                --lam_sdiv $lam_sdiv                    \
                --lam_sfn $lam_sfn                      \
                --lam_grads 0                           \
                --lam_curl $lam_curl                    \
                --lam_pde 0                             \
                --lam_weight 1e-4                       \
                --sfn_num_centers 666                   \
                --sfn_num_increments $ni                \
                --sfn_patch_dim   666                   \
                --gpus 0 --max_epochs $max_epoch        \
                --log_every_n_steps 10 --check_val_every_n_epoch 20

done
done
done
done
done
done
done
done
done