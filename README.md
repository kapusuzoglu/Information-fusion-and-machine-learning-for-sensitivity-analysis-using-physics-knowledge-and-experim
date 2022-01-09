# Information-fusion-and-machine-learning-for-sensitivity-analysis-using-physics-knowledge-and-experimental data

## Highlights 

• Physics-informed machine learning is investigated for global sensitivity analysis.  
• Physics and test data are fused to maximize the accuracy of sensitivity estimates.  
• Uncertainties in Gaussian process and deep neural network models are included.  
• Accuracy, uncertainty and computational effort of proposed approaches are compared.

## Abstract

When computational models (either physics-based or data-driven) are used for the sensitivity analysis of engineering systems, the sensitivity estimate is affected by the accuracy and uncertainty of the model. This paper considers global sensitivity analysis (GSA) for situations where both a physics-based model and experimental observations are available, and investigates physics-informed machine learning strategies to effectively combine the two sources of information in order to maximize the accuracy of the sensitivity estimate. Two representative machine learning (ML) techniques are considered, namely, deep neural networks (DNN) and Gaussian process (GP) modeling, and two strategies for incorporating physics knowledge within these techniques are investigated, namely: (i) incorporating loss functions in the ML models to enforce physics constraints, and (ii) pre-training and updating the ML model using simulation and experimental data respectively. Four different models are built for each type (DNN and GP), and the uncertainties in these models are included in the Sobol’ indices computation. The DNN-based models, with many degrees of freedom in terms of model parameters and training options, are found to result in smaller bounds on the sensitivity estimates when compared to the GP-based models. The proposed methods are illustrated for additive manufacturing and lake temperature modeling examples.

## Cite Paper
Kapusuzoglu, B., & Mahadevan, S. (2021). Information fusion and machine learning for sensitivity analysis using physics knowledge and experimental data. Reliability Engineering & System Safety, 214, 107712.

Please, cite this repository using: 

    @article{kapusuzoglu2021information,
      title={Information fusion and machine learning for sensitivity analysis using physics knowledge and experimental data},
      author={Kapusuzoglu, Berkcan and Mahadevan, Sankaran},
      journal={Reliability Engineering \& System Safety},
      volume={214},
      pages={107712},
      year={2021},
      doi={10.1016/j.ress.2021.107712},
      publisher={Elsevier}
    }
