# dl4l-cog

Try to investigate emotion in connection with other variables, like age, gender, psychological state (e.g., depression), political view, ... One can train an emotion classifier, and assign emotions to texts marked for other variables as above. There are annotated corpora for most of these (and more), or it would be easier to obtain through publicly available data.
Trying to answer questions like:
- How does happiness change over age?
- Do populist politicians more angry, or joyful in their speeches?
- What emotion correlates with conditions like depression, PTSD etc, anger, sadness?

huggingface search - classifier
https://huggingface.co/models?pipeline_tag=text-classification&sort=downloads&search=emotion


## MENTAL HEALTH DATASETS
https://github.com/kharrigian/mental-health-datasets

e.g. SMHD (Self-reported Mental Health Diagnoses) Dataset
https://arxiv.org/pdf/1806.05258v2.pdf <- paper, which introduces the dataset
https://ir.cs.georgetown.edu/resources/ <- lab website also with more datasets

Unfortunately as stated in data request form (https://docs.google.com/forms/d/e/1FAIpQLScC-O3MXDd2lZSGqeRHsv1EMVR2xN5WC0cAodsHK3tBOz_FLw/viewform):
"The datasets are for research purposes only and cannot be made available for class projects or commercial use." 
-> How do we deal with this?

## POLITICAL DATASETS
https://www.kaggle.com/datasets/manchunhui/us-election-2020-tweets
https://huggingface.co/datasets/hugginglearners/russia-ukraine-conflict-articles

## CONVERSATION DATASETS
How do emotions change in social interactions after a particularly emotion-loaded comment?
https://www.kaggle.com/datasets/nltkdata/nps-chat
http://www.psych.ualberta.ca/~westburylab/downloads/usenetcorpus.download.html
http://dataset.cs.mcgill.ca/ubuntu-corpus-1.0/

### 4chan /pol/ Dataset
Papasavva et al. Raiders of the Lost Kek: 3.5 Years of Augmented 4chan Posts from the Politically Incorrect Board. (2020). 14th International AAAI Conference On Web And Social Media (ICWSM), 2020.
https://arxiv.org/abs/2001.07487
https://zenodo.org/record/3606810
&rarr; freely available!

<!-- @article{papasavva2020raiders,
  title={Raiders of the Lost Kek: 3.5 Years of Augmented 4chan Posts from the Politically Incorrect Board},
  author={Antonis Papasavva, Savvas Zannettou, Emiliano De Cristofaro, Gianluca Stringhini, Jeremy Blackburn},
  journal={14th International AAAI Conference On Web And Social Media (ICWSM), 2020},
  year={2020}  
} -->
