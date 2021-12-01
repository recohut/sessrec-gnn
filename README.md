
# Session-based Recommendation with Graph Neural Networks

## Tree
```
.
├── [ 68M]  data
│   ├── [ 58M]  diginetica
│   │   ├── [ 17M]  processed
│   │   │   ├── [1.5M]  test.txt
│   │   │   └── [ 16M]  train.txt
│   │   └── [ 41M]  raw
│   │       └── [ 41M]  train-item-views.csv
│   ├── [594K]  sample
│   │   ├── [200K]  processed
│   │   │   ├── [   4]  num_items.txt
│   │   │   ├── [ 23K]  test.txt
│   │   │   └── [173K]  train.txt
│   │   └── [390K]  raw
│   │       └── [386K]  sample_train-item-views.csv
│   └── [9.5M]  yoochoose1_64
│       └── [9.5M]  processed
│           ├── [1.2M]  test.txt
│           └── [8.2M]  train.txt
├── [608K]  notebooks
│   ├── [ 17K]  analyzing_session_graph_with_attention_in_pytorch.ipynb
│   ├── [ 88K]  convert_session_to_session_graph_yoochoose_dataset.ipynb
│   ├── [ 37K]  fgnn_session_based_recommendation_on_sample_dataset.ipynb
│   ├── [ 61K]  gce_gnn_session_based_recommendation_on_diginetica_dataset.ipynb
│   ├── [ 62K]  gce_gnn_session_based_recommendation_on_nowplaying_dataset.ipynb
│   ├── [ 62K]  gce_gnn_session_based_recommendation_on_tmall_dataset.ipynb
│   ├── [ 31K]  gc_san_session_recommendation_model_pytorch.ipynb
│   ├── [ 40K]  lessr_session_based_recommendation_on_sample_dataset.ipynb
│   ├── [ 23K]  preprocessing_diginetica.ipynb
│   ├── [ 13K]  preprocessing_gowalla.ipynb
│   ├── [ 14K]  preprocessing_lastfm.ipynb
│   ├── [ 15K]  preprocessing_sample_data.ipynb
│   ├── [ 37K]  srgnn_session_based_recommendation_on_sample_dataset.ipynb
│   ├── [ 41K]  tagnn_pp_session_based_recommendation_on_diginetica_dataset.ipynb
│   ├── [ 27K]  tagnn_pp_session_based_recommendation_on_yoochoose_64_dataset.ipynb
│   └── [ 38K]  tagnn_session_based_recommendation_on_sample_dataset.ipynb
├── [2.1K]  README.md
├── [998K]  report
│   ├── [932K]  images
│   │   ├── [ 20K]  Untitled 10.png
│   │   ├── [ 28K]  Untitled 11.png
│   │   ├── [ 11K]  Untitled 12.png
│   │   ├── [ 28K]  Untitled 13.png
│   │   ├── [ 35K]  Untitled 14.png
│   │   ├── [ 28K]  Untitled 15.png
│   │   ├── [126K]  Untitled 1.png
│   │   ├── [ 81K]  Untitled 2.png
│   │   ├── [103K]  Untitled 3.png
│   │   ├── [144K]  Untitled 4.png
│   │   ├── [ 89K]  Untitled 5.png
│   │   ├── [ 95K]  Untitled 6.png
│   │   ├── [ 45K]  Untitled 7.png
│   │   ├── [9.9K]  Untitled 8.png
│   │   ├── [ 20K]  Untitled 9.png
│   │   └── [ 64K]  Untitled.png
│   └── [ 62K]  sessionrec_gnn_report.html
└── [745K]  scripts
    ├── [184K]  dgtn
    │   ├── [ 31K]  data
    │   │   ├── [   0]  __init__.py
    │   │   ├── [7.8K]  multi_sess_graph.py
    │   │   └── [ 20K]  __pycache__
    │   │       ├── [ 123]  __init__.cpython-36.pyc
    │   │       └── [ 15K]  multi_sess_graph.cpython-36.pyc
    │   ├── [6.7K]  main.py
    │   ├── [ 82K]  model
    │   │   ├── [3.3K]  ggnn.py
    │   │   ├── [   0]  __init__.py
    │   │   ├── [ 13K]  InOutGat.py
    │   │   ├── [ 13K]  model.py
    │   │   ├── [7.0K]  multi_sess.py
    │   │   ├── [ 37K]  __pycache__
    │   │   │   ├── [3.4K]  ggnn.cpython-36.pyc
    │   │   │   ├── [ 126]  __init__.cpython-36.pyc
    │   │   │   ├── [ 10K]  InOutGat.cpython-36.pyc
    │   │   │   ├── [ 11K]  model.cpython-36.pyc
    │   │   │   ├── [5.0K]  multi_sess.cpython-36.pyc
    │   │   │   └── [3.6K]  srgnn.cpython-36.pyc
    │   │   └── [4.5K]  srgnn.py
    │   ├── [ 16K]  neigh_retrieval
    │   │   ├── [   0]  __init__.py
    │   │   ├── [3.7K]  knn.py
    │   │   ├── [1.2K]  neighborhood_retrieval.py
    │   │   └── [7.5K]  __pycache__
    │   │       ├── [ 145]  __init__.cpython-36.pyc
    │   │       └── [3.3K]  knn.cpython-36.pyc
    │   ├── [ 28K]  preprocess
    │   │   ├── [6.0K]  cikm16_org_prepro.py
    │   │   ├── [5.5K]  cikm16_perprocess.py
    │   │   ├── [6.3K]  rcs15_org_prepro.py
    │   │   └── [6.5K]  rcs15_perprocess.py
    │   ├── [ 968]  README.md
    │   ├── [4.3K]  train.py
    │   └── [9.5K]  utils
    │       ├── [   0]  __init__.py
    │       ├── [4.8K]  __pycache__
    │       │   ├── [ 126]  __init__.cpython-36.pyc
    │       │   └── [ 713]  saver.cpython-36.pyc
    │       └── [ 646]  saver.py
    ├── [ 25K]  fgnn
    │   ├── [3.3K]  dataset.py
    │   ├── [3.3K]  gru_set2set.py
    │   ├── [4.0K]  main.py
    │   ├── [3.3K]  model.py
    │   ├── [ 314]  README.md
    │   ├── [1.6K]  train.py
    │   └── [4.8K]  weighted_gat.py
    ├── [ 26K]  gcegnn
    │   ├── [3.6K]  aggregator.py
    │   ├── [1.5K]  build_graph.py
    │   ├── [4.2K]  main.py
    │   ├── [8.3K]  model.py
    │   ├── [1.1K]  README.md
    │   └── [3.5K]  utils.py
    ├── [ 40K]  lessr
    │   ├── [6.4K]  lessr.py
    │   ├── [3.3K]  main.py
    │   ├── [ 303]  packages.yml
    │   ├── [1.5K]  preprocess.py
    │   ├── [3.2K]  README.md
    │   └── [ 21K]  utils
    │       ├── [ 13K]  data
    │       │   ├── [1.5K]  collate.py
    │       │   ├── [1.5K]  dataset.py
    │       │   └── [5.7K]  preprocess.py
    │       └── [4.4K]  train.py
    ├── [7.4K]  preprocess.py
    ├── [ 46K]  srgnn
    │   ├── [ 17K]  pytorch_code
    │   │   ├── [3.3K]  main.py
    │   │   ├── [6.1K]  model.py
    │   │   └── [3.9K]  utils.py
    │   ├── [4.5K]  README.md
    │   └── [ 20K]  tensorflow_code
    │       ├── [4.1K]  main.py
    │       ├── [7.2K]  model.py
    │       └── [5.2K]  utils.py
    ├── [ 18K]  tagnn
    │   ├── [3.2K]  main.py
    │   ├── [6.9K]  model.py
    │   ├── [ 127]  README.md
    │   └── [3.8K]  utils.py
    ├── [385K]  tagnn_pp
    │   ├── [3.4K]  agc.py
    │   ├── [361K]  assets
    │   │   ├── [   1]  new
    │   │   ├── [151K]  Results_plot.png
    │   │   ├── [153K]  SBR_Task.png
    │   │   └── [ 53K]  TAGNN++.png
    │   ├── [8.0K]  model.py
    │   ├── [3.3K]  proc_utils.py
    │   └── [4.9K]  train.py
    └── [9.8K]  time_preprocessing.py

  70M used in 34 directories, 118 files
```