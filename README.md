# iMet Collection 2019 - FGVC6

## 1st place
- RandomCropIfNeeded
- batch accumulation
- With SIZE = 320 for SEResNext101 / SENet154 and SIZE = 331 for PNasNet-5
- Filtering predictions by high loss
- Pseudo labeling add the most confident predictions (highest np.mean(np.abs(probabilities - 0.5)) ) to the training dataset.
- Culture and tags separately
- 2nd stage I construct the binary classification dataset: I took 1103 (number of classes) rows per each image and trying to predict that this class relates to this image (0 or 1)
- feature engineering 
- probabilities of each models, sum / division / multiplication of each pair / triple / .. of models
- mean / median / std / max / min of each channel
- brightness / colorness of each image (you can say me that NN can easily detect it — yes, but here i can do it without cropping and resizing — it is less noisy)
- Max side size and binary flag — height more than width or no (it is a little bit better for tree boosting than just height + width in case of lower side == 300)
- add all 1000 (number of ImageNet classes) predictions to this dataset
- lightgbm
- postprocess
- (Done)Different threshold for cultures and tags models

## 2nd place
- The technique was quite similar to my quickdraw solution. I put top 30 tags and top 20 cultures to the dataset for LGMB

## 4th place
- Tag Relevance Prediction

## 6th place
-  (Done)adjust the threshold for each image according to the max probability of that image

## 9th place
- knowledge Distillation
- train only fresh params

## 10th place
- Small modifications were made to the network, our logit takes input from the last two layers rather than the last one layer

## others
- two a fully connected layers, one for the culture and one for the tag



## Usage

### Train
Make folds
```
python make_folds.py --n-folds 40
```

Train se_resnext101 from fold 0 to 9:
```
python main.py train model_se101_{fold} --model se_resnext101_32x4d --fold {fold} --n-epochs 40 --batch-size 32 --workers 8
```
Train inceptionresnetv2 from fold 5 to 9:
```
python main.py train model_inres2_{fold} --model inceptionresnetv2 --fold {fold} --n-epochs 40 --batch-size 32 --workers 8
```
Train pnas models from fold 0 to 4:
```
python main.py train model_pnas_{fold} --model pnasnet5large --fold {fold} --n-epochs 40 --batch-size 24 --workers 8
```
### Test
The ensemble of these model is used to predict results in `imet-predict-final.ipynb`.
