## Neural-based Noise Filtering from Word Embeddings
Kim Anh Nguyen, nguyenkh@ims.uni-stuttgart.de

Code for paper [Neural-based Noise Filtering from Word Embeddings](http://www.ims.uni-stuttgart.de/institut/mitarbeiter/anhnk/papers/coling2016/denoising-embeddings.pdf) (COLING 2016).

### Requirements
  1. [SPAMS toolbox with python interface](http://spams-devel.gforge.inria.fr)
  2. Theano
  
### Pre-trained word embeddings
  - The models can filter noise from any pre-trained word embeddings such as word2vec, GloVe
  - The format of word embeddings used in this code is format of word2vec (either binary or text)
  
### Preprocessing
  - This step is to learn the dictionaries for CompEmb and OverCompEmb models; transform complete word embeddings to overcomplete word embeddings.
  - Running command:
  
    ```python preprocessing.py -input <original_embs_file> -output <overcomp_file> -factor <factor_overcomplete>```
    
    For example, transform an input word embeddings of 100 dimensions into overcomplete word embeddings of 1000 dimensions (factor == 10):
  
    ```python preprocessing.py -input sgns_100d.bin -output sgns_overcomp_1000d.bin -factor 10```
    
### Training models
  1. Training CompEmb model:
  
    ```THEANO_FLAGS="mode=FAST_RUN,device=cpu,floatX=float32" python filter_noise_embs.py -input sgns_100d.bin -output sgns_denoising_100d.bin -iter 30 -bsize 100```
    
    Train CompEmb model with 30 iterations and batch size of 100.
  
  2. Training OverCompEmb model:
  
    ```THEANO_FLAGS="mode=FAST_RUN,device=cpu,floatX=float32" python filter_noise_embs.py -input sgns_100d.bin -output sgns_denoising_1000d.bin -over sgns_overcomp_1000d.bin -iter 30 -bsize 100```

  Train OverCompEmb model with 30 iterations and batch size of 100; sgns_overcomp_1000d.bin is an overcomplete word embeddings.
 
### Reference
```
@InProceedings{nguyen:2016:denoising
  author    = {Nguyen, Kim Anh and Schulte im Walde, Sabine and Vu, Ngoc Thang},
  title     = {Neural-base Noise Filtering from Word Embeddings},
  booktitle = {Proceedings of the 26th International Conference on Computational Linguistics (COLING)},
  year      = {2016},
  address = {Osaka, Japan},
}
```
