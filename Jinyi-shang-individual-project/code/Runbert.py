# The following code is running on the Colaboratory. Since google provide a free Gpu and it's faster to run deep learning model
# Here is the link: https://colab.research.google.com/notebooks/gpu.ipynb
# Notice, if you are the first user of colab, remember to authorize
# As for bert tutorial:
# first, you need to download the the pretrain bert model: https://github.com/google-research/bert
# Then, upload your file to the google drive, open a new ipynb


# authorization:
# !apt-get install -y -qq software-properties-common python-software-properties module-init-tools
# !add-apt-repository -y ppa:alessandro-strada/ppa 2>&1 > /dev/null
# !apt-get update -qq 2>&1 > /dev/null
# !apt-get -y install -qq google-drive-ocamlfuse fuse
# from google.colab import auth
# auth.authenticate_user()
# from oauth2client.client import GoogleCredentials
# creds = GoogleCredentials.get_application_default()
# import getpass
# !google-drive-ocamlfuse -headless -id={creds.client_id} -secret={creds.client_secret} < /dev/null 2>&1 | grep URL
# vcode = getpass.getpass()
# !echo {vcode} | google-drive-ocamlfuse -headless -id={creds.client_id} -secret={creds.client_secret}



# set your envirnoment:
from google.colab import drive
drive.mount('/content/drive/')

# cd /content/drive//My\ Drive/bert-master


# run this model

# !python run_classifier.py \
#   --task_name=cola \
#   --do_train=true \
#   --do_eval=true \
#   --do_predict=true \
#   --data_dir=6103project \
#   --vocab_file=cased_L-12_H-768_A-12/vocab.txt \
#   --bert_config_file=cased_L-12_H-768_A-12/bert_config.json \
#   --init_checkpoint=cased_L-12_H-768_A-12/bert_model.ckpt \
#   --max_seq_length=40 \
#   --train_batch_size=8 \
#   --learning_rate=2e-5 \
#   --num_train_epochs=3.0 \
#   --output_dir=output \
#   --do_lower_case=False

# Here, we should notice, tensorflow 2.2 may not run bert model, so you can change your tensorflow version
# !pip install tensorflow==1.15