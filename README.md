# Dien----Deep-Interest-Evolution-Network-for-Click-Through-Rate-Prediction
An implementation of Paper "Deep Interest Evolution Network for Click-Through Rate Prediction"
The paper url is: https://arxiv.org/abs/1809.03672 
The official implementation github is: https://github.com/mouna99/dien

Environment:
TF-2.7.0
Cuda 11.0
3090x2

This is a simplified version of DIEN, some experimental parts in offical codes are ignored. negative sampling is optimized and got very large speed up.
Running result is shown on report.txt
Dataset used is Amazon dataset, version:2018 category:Book review:core-5 

I just wanna see the implementation of DIN initially, then the offical git guide me to DIEN.
I'm not a guy who like to upload my learning procedure to git, but this project, which needs to reimplement a GRU module and a lot of other work paid me around 5 days.
Then I think maybe my implementation could help some other people. You could check the model part without a lot of messy tf 1.x code in official implementation.

There are some problems are commented in codes. 

If you find something wrong, please open an issue and tell me.
This repository is not intent to teach other something, I'd like someone can point out my problem and learn something.
Thanks.
