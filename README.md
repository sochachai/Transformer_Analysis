# Goal
This project aims to build a text decoder by constructing the components, e.g. encoders, decoders, multihead attentions etc. of a transformer model to solve the a text decryption problem described in more details as follows. 
# Problem Description
Assume a collection of English texts is decoded by a one-to-one map or a permutation from the set of English alphabet, i.e. {a-z} to itself, with other none-letter symbols being unchanged. <br />
For example, {a,b,c,d,e,...,x,y,z} is decoded to {f,g,h,i,j,...,c,d,e}, i.e. each letter is moved forward with 5 positions, with the last few letters going back to the front. The original text "he is a boy" is then decoded to "mj nx f gtd". \n
We will call the above example as "5-plus-shift" later on.<br />
Another example is {a,b,c,d,e,...} is decoded to {g,z,k,w,b,...} as long as every letter is decoded to a unique letter and no two letters share a same decoded image.<br />
The task is to train a language model to find the decoding pattern so when prompted a masked text, say the "mj nx f gtd" example above, the model can translate it to "he is a boy"
# Data
Some English texts along with their "5-plus-shift" decoded texts as training sets and validation sets for the machine to learn. Test sets are from user inputs(see more on the App section).<br /> The data has been processed such that 1.each line of the text consists of at most 40 characters including letters and symbols; <br />2.all English letters are in lower-case.
# Model & Method
Each component of the transformer model is written from scratch in the sense that the transformer model is not applications of pre-trained LLM models but it does rely on Pytorch for the construction of neural networks.
The model is trained with Google Colab. The trained model is saved in dictionary format which could be loaded in other Python environments, say on-premise local machines or VMs in Google Cloud Platform, using the same dictionary format.  
The alphabet is tokenized with {1,2,...,26}. Other symbols that are not letters are all tokenized with the number 0.
# App Deployment
#### 1.The app represents itself as a web application where the user can enter one sentence of masked text and the web app will return the original English text upon the user's Submit request.<br />
#### 2.The app is hosted with App Engine in Google Cloud Platform which can be triggered using VMs.<br />
  To set this up, <br />
  2.1. Store scripts main.py(the app run), language.py(the transformer code for which main.py will call), my_transformer_utils(a dependency package of language.py), the pickle file letter_decryption_core_v1.pkl(the model trained in Google Lab), app.yaml and requirements.txt(dependencies for Cloud App Build) into a Google Cloud Store Bucket.<br />
  2.2. Select and open a VM instance with machine type:e2-custom-4-2048 or any other configuration with sufficient memory, disk space and compute power. The free configuration would not work in our case as the creator has tested.
  2.3. SSH the VM and open a virtual environment in the SSH terminal with "python3 -m venv .venv".<br />
  2.4. Activate the virtual enviroment by typing "source .venv/bin/activate" in the SSH terminal and use "which python" for validation.<br />
  2.5. Transport the project files/folders described in Step 2.1 to the virtual environment described in Step 2.3 using commands "gsutil cp ..." or "gsutil cp -r". <br />
  2.6. Install dependencies of this project in the virtual environment by typing "python3 -m pip install torch" etc. Use "python3 -m pip install --upgrade pip" to install "pip" if necessary. <br />
  2.7. Test the model in SSH terminal with "Python3 language.py". If successed, the message "This is an apple." should display itself. <br />
  2.8. Type "gcloud auth login" and go to the prompted page to finish authorization process. <br />
  2.9. Go to IAM & ADMIN in Google Cloud Console to add the "...-compute@developer.gserviceaccount.com" as an APP ENGINE DEVELOPER.<br />
  2.9. Type "gcloud config set app/cloud_build_timeout 7200s" to set a time-out limit for app deployment (note this is not a hello-world-level flask cloud app deployment).<br />
  2.10. Type "nano app.yaml" and "nano requirements.txt" to properly configure the cloud build settings(a low memory_gb/a low disk_size_gb in app.yaml or missing torch will result in early undesired termination of the app deployment).<br />
  2.11. Type "gcloud app deploy" for the app to deploy in Google Cloud. <br />
  2.12. Wait for the app deployment process to finish then go to the URL provided in the SSH terminal to use the web app.<br />
# Result
The web app successfully finds the "5-plus-shift" pattern and can translate a masked text to its original English text.
Since the web application established in Google Cloud incurs fees, I have terminated the process but instead attached a few screen shots for result demonstrations.
  
  
# Reference: 
The famous paper "Attention is All You Need" 
# Thanks to: 
1. Stack overflow help links on Cloud deployment such as https://stackoverflow.com/questions/65768595/gcloud-app-deploy-fails-cloud-build-did-not-succeed-within-10m; https://stackoverflow.com/questions/51025893/flask-at-first-run-do-not-use-the-development-server-in-a-production-environmen etc.
2. Transformer code instructions such as https://www.youtube.com/watch?v=U0s0f995w14; https://www.bilibili.com/video/BV1kT4y1b7us/?spm_id_from=333.337.search-card.all.click etc.


