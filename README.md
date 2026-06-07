# Bachelors Thesis Codebase

GitHub code base for the Bachelors Thesis:

Is Free Will Compatible With Determinism? A Psychometric Study Of Large Language Models

The Free Will Inventory by Nadelhoffer et al.:
* https://www.sciencedirect.com/science/article/abs/pii/S1053810014000075

Vendor API documentation:
* https://api-docs.deepseek.com/
* https://developers.openai.com/api/docs
* https://docs.mistral.ai/
* https://platform.claude.com/docs/en/home
* https://ai.google.dev/gemini-api/docs
* https://grok-api.apidog.io/
* https://lmstudio.ai/docs/python

# Requirements:
All code tested using Python 3.12.9.

## Python Packages
* openai
* google-genai
* anthropic
* mistralai
* lmstudio
* xai-sdk
* pandas==2.1.0
* numpy==1.25.2
* matplotlib==3.6.3
* scipy==1.12.0
* ridgeplot
* plotly
* kaleido
* seaborn

## R packages
* jsonlite
* PlackettLuce
* qvcalc
* dplyr

# Directory
```text                              
FreeWill-LLMs/                              
├── deprecated/               # Old files, can be ignored
│                        
├── drawio_files/             # .drawio files for all figures
├── Plotting/                 # Scripts and csv's related to plotting
│                        
├── rankings_json_part1/      # All JSON ranking data for Part I
├── rankings_json_part2/      # All JSON ranking data for Part II           
├── results_FC_part1/         # Results from Forced-Choice Part I
├── results_FC_part2/         # Results from Forced-Choice Part II         
├── results_SDS/              # Results from Social Desirability Scoring
├── results_tables/           # Data for Dimension Score Tables
│                        
├── FWI_Part1_and_Part2.csv   # The Free Will Inventory by Nadelhoffer et al.             
├── json_writer.py            # Convert results_FC_partX -> rankings_json_partX
├── model_runner.py           # Coordinates API calls to various providers
├── structures.py             # Dataclasses to store results, also store FWI
│                        
├── part1_block_design.py     # Part I, block optimizer      
├── part1_prompting.py        # Part I, Forced-Choice Prompting           
├── part1_scoring.py          # Part I, dimension scores and bootstrap          
├── part1_SDS.py              # Part I, SDS scores
│                        
├── part2_block_design.py     # Part II, block optimizer                 
├── part2_prompting.py        # Part II, Forced-Choice Prompting
├── part2_scoring.py          # Part II, dimension scores and bootstrap
├── part2_SDS.py              # Part II, SDS scores       
│                        
├── pkl_fixer.ipynb           # Manual fixer for ranking errors            
├── R_contrasts_bootstrap.r   # R -> to calculate worth contrast and bootstrap                    
└── R_statistics_pooling.r    # R -> to calculate pooled worth, bootstrap and GLHT tests
 
```
