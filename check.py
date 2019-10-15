from tensorflow.contrib.framework.python.framework import checkpoint_utils
var_list = checkpoint_utils.list_variables("./model.ckpt")
for v in var_list:
    print(v)

