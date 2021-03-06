# visual-reasoning-rationalization

Code associated with the "Natural Language Rationales with Full-Stack Visual Reasoning" Findings of EMNLP 2020 paper

## Citation

```
@inproceedings{marasovic-et-al-2020-rationalization,
    title = "{{Natural Language Rationales with Full-Stack Visual Reasoning: From Pixels to Semantic Frames to Commonsense Graphs}}",
    author = "Marasovi\'{c}, Ana and Bhagavatula, Chandra and Park, Jae Sung and Le Bras, Ronan and Smith, Noah A. and Choi, Yejin",
    booktitle = "Findings of EMNLP",
    year = "2020",
    url = "https://arxiv.org/abs/2010.07526"
}
```

## Installation 

```
conda env create -f environment.yml
conda activate rationalization
```

## Trained models 

https://visual-reasoning-rationalization.s3-us-west-2.amazonaws.com/models_vqa.zip

https://visual-reasoning-rationalization.s3-us-west-2.amazonaws.com/models_vcr.zip

https://visual-reasoning-rationalization.s3-us-west-2.amazonaws.com/models_e_snli_ve.zip

Extract in an empty directory named `models` using `tar -xzvf`. 

## Downloading data 

https://visual-reasoning-rationalization.s3-us-west-2.amazonaws.com/data.zip

## Download features 

https://visual-reasoning-rationalization.s3-us-west-2.amazonaws.com/features.zip

## Example commands 

### Training

```bash
export FEATURES=textual_objects
python scripts/run_finetuning.py -e ${FEATURES} 
```

The value of `FEATURES` can be one of the following: `text_only, textual_objects, embedding_objects, textual_situation, embedding_situation, textual_viscomet, textual_viscomet`.


### Decoding  

```bash
python scripts/run_generation.py -e ${FEATURES}_eval --model_name_or_path /models/vcr_gen/q_a_to_r/
```


## Templates for human evaluation

HTML files can be found at `human_eval_templates`
