# Neural Sentence Ordering

An python (with theano) based implementation of Paper "Neural sentence ordering".
Any usage of our code, data or idea could cite the paper "Neural sentence ordering" on arXiv .

## Prepare

- Python 2.7
- Cuda with cudnn v5
- GCC < v5

Then, execute

```sh
source env.sh
pip install -r requirements.txt
edit src/.theanorc # set theano configurations, especially cuda.root
```

## Data

The data is available on http://nlp.fudan.edu.cn/data/.

The sample of processed data could be found on https://drive.google.com/file/d/0B-mnK8kniGAiSWhaR3gyalJyQm8/view?usp=sharing. And users should put this *.gz file into ./data/ to run the code.
This processed data is based only a toy way to organize data in order to make the code run.

### Format

- **data/\*.pkl.gz**: cPickled, gzipped tuple `data`

  - **data**: tuple(`src_train`, `src_valid`, `src_test`, `dic_w2idx`, `dic_idx2w`, `dic_w2embed`, `embedding`)
  
    - **src_\***: list<tuple(`paragraph`, `categories`)>

      - **paragraph**: list<`sentence`>
        - **sentence**: list<`idx`>
      - **categories**: list\<str\>
      
    - **dic_w2idx**: dict<`word`: `idx`>
    
    - **dic_idx2w**: dict<`idx`: `word`>
    
    - **dic_w2embed**: dict<`word`: `embed`>
    
      - **embed**: `np.ndarray(shape=(wordVecLen, ), dtype='float64')`
    
    - **embedding**: `np.ndarray(shape=(vocabSize, wordVecLen), dtype='float64')`

  - **word**: str, **idx**: int
      

## Run

The entrance of the code is ./src/driver.py
