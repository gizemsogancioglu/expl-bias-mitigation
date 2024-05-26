# Using Explainability for Bias Mitigation: A Case Study for Fair Recruitment Assessment
[![Size](https://img.shields.io/github/repo-size/gizemsogancioglu/expl-bias-mitigation)](https://img.shields.io/github/repo-size/gizemsogancioglu/expl-bias-mitigation)
[![License](https://img.shields.io/github/license/gizemsogancioglu/expl-bias-mitigation)](https://img.shields.io/github/license/gizemsogancioglu/expl-bias-mitigation)
![GitHub top language](https://img.shields.io/github/languages/top/gizemsogancioglu/expl-bias-mitigation)

Here, we provide source code to ensure reproducibility for the following paper: [Using Explainability for Bias Mitigation: A Case Study for Fair Recruitment Assessment, Gizem Sogancioglu, Heysem Kaya, Albert Ali Salah]

        .
        ├── source                              # including all source files                 
        │   ├── bias_measures.py                # bias measures (Equal Accuracy and PCC)
        │   ├── training.py                     # training of blackbox models. 
        │   ├── data_loader.py                  # load dataset of FairCVdb and prepare features.
        │   ├── proxyMute.py                    # proxyMute algorithm implementation
        │   ├── exp.py                          # main script processing all steps 
        ├── faircvtest                         
        │   ├── Profiles_{train/test}_gen.npy   # locate the files here (download here: https://github.com/BiDAlab/FairCVtest) 
        └── results                             # output files reporting performance and fairness measures of ProxyMute algorithm. 

## ProxyMute algorithm pipeline

![Alt text](pipeline.png?raw=true "The proposed bias mitigation pipeline (ProxyMute) using feature attribution-based explainability method.")

## References
* Paper: [Using Explainability for Bias Mitigation: A Case Study for Fair Recruitment Assessment,Gizem Sogancioglu, Heysem Kaya, Albert Ali Salah]
* For more information or any problems, please contact: gizemsogancioglu@gmail.com
