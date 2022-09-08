Dynamic Graph-Level Neural Network for SAR Image Change Detection
Rongfang Wang, Liang Wang, Xiaohui Wei, Jia-Wei Chen, and Licheng Jiao

This is the pytorch implementionof the DGLNN 

The training data is in the GCN-data/train folder, and test data is in the GCN-data/test folder

We provide several trained models of DGLNN for evaluation in models_save floder

ref_img folder contains ground truth of each datasets 

cd_data.py is used to load data

config.py consists some parameters in the training process.

The DGNN model is defined in the models_knn.py
 
you can use the run_knn.py to obtain the CD result:
python run_knn.py

testresults.py is a function to compute the kappa FP and MP

If you find this idea or code useful for your research, please consider citing our paper:

@article{wang2021dynamic,
  title={Dynamic graph-level neural network for SAR image change detection},
  author={Wang, Rongfang and Wang, Liang and Wei, Xiaohui and Chen, Jia-Wei and Jiao, Licheng},
  journal={IEEE Geoscience and Remote Sensing Letters},
  volume={19},
  pages={1--5},
  year={2021},
  publisher={IEEE}
}

