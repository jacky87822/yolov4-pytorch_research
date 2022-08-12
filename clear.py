import os

path ="./result/"

dir=os.listdir(path)
for i in dir:
    if "backup" not in i:
        try:
            sub_dir=os.listdir(os.path.join(path,i))
            for j in sub_dir:
                cur_path=os.path.join(path,i,j)
                print (cur_path)
                os.system ("rm -rf "+cur_path+"/*.json")
                os.system ("rm -rf "+cur_path+"/inj_list*")
        except:
            pass
