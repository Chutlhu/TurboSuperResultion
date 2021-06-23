import subprocess

path_to_convert =  "convert"

# import matplotlib.pyplot as plt
# from matplotlib import rcParams
# rcParams['savefig.transparent'] = True
# rcParams['savefig.dpi'] = 130
# rcParams['savefig.pad_inches'] = 0
# plot_args = {'rstride': 1, 'cstride': 1, 'cmap':"RdYlBu",
#              'linewidth': 0.5, 'antialiased': True, 'color': '#1e1e1e',
#              'shade': True, 'alpha': 1.0, 'vmin': -1, 'vmax':1}


def execute_backgraund(cmd):
    args = [path_to_convert, "-delay", "10", "-loop" , "0", "-dispose", "Background", "%s*.%s" % (path_to_imgs, img_ext), outputfile]
    print(args)
    subprocess.call(args, shell=True)

def execute_and_print(cmd):
    print('here')
    popen = subprocess.Popen(cmd, stdout=subprocess.PIPE, universal_newlines=True)
    print('here')
    for stdout_line in iter(popen.stdout.readline, ""):
        yield stdout_line 
    popen.stdout.close()
    return_code = popen.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, cmd)


def make_animation(outputfile, path_to_imgs, img_ext):
    cmd = [path_to_convert, "-delay", "10", "-loop" , "0", "-dispose", "Background", "%s*.%s" % (path_to_imgs, img_ext), outputfile]
    execute_and_print(cmd)
    # cmd2 = (["del", "/Q",  "%s*.%s" % (path_to_imgs, img_ext)], shell=True)
    # print("\ndone")
    pass




# r, t = mgrid[0:1:20j, 0:2*pi:40j]
# x, y = r*cos(t), r*sin(t)
# fig = plt.figure(facecolor=None, frameon=False)
# ax = fig.add_subplot(111, projection='3d')
# for i in range(30):
#     data_gen(i)
#     plt.savefig("drum_{n:02d}.png".format(n=i), transparent=True,  frameon=False)
#     print i, 

