# Using Explainability for Bias Mitigation: A Case Study for Fair Recruitment Assessment
Here, we provide source code to ensure reproducability for the following paper: [Using Explainability for Bias Mitigation: A Case Study for Fair Recruitment Assessment,Gizem Sogancioglu, Heysem Kaya, Albert Ali Salah]

- Extracted features from the videos of ChaLearn LAP-FI dataset (features/feature_[train/test/validation].csv)  
- Scripts for mood and personality prediction (source/*.py)  

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
