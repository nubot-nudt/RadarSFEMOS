from main import main

class IOStream:
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text + '\n')
        self.f.flush()

    def close(self):
        self.f.close()


textio = IOStream('param_sigma_opt.log')
param_name = 'sigma_rrv'
param_ls = [0.1,0.15,0.2,0.25,0.3,0.35,0.4]
best_val_ls = []
for param_v in param_ls:
    best_val_res = main(param_name,param_v)
    best_val_ls.append(best_val_res)

textio.cprint('====The result for param tuning====')
for i in range(len(param_ls)):
    textio.cprint(str(param_ls[i]) + ':' + str(best_val_ls[i])[:6])
