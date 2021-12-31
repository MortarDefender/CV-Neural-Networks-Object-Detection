### Task To Do:

## Part 1:
# reaserch Mobilenet SSD
- loss function
- layers that are not [linear, conv, pool]
- other functionality added in the model

## Part 2:
# choose the object to detect
# get a large dataset for object detecting
# learn how to use Mobilenet SSD and how to freeze and remove the last layers of the model
# start tuneing the model
# try to freeze all but the last layers and try it on a diffrent smaller data set

## build:
# GetDataSets -> (class) will scrap or get from a website the needed images
                        will ogment a few images to create new images [rotation, sher, translation, color, filters...]
# NNModel -> (class) includes an import (or the code) for Mobilenet SSD with the tuneing required
# TransferLearning -> (class) transfer learning from the model in NNModel to a diffrent data set (using GetDataSet)
# Main -> (file) -> that includes a main function that 
                    can use the NNModel for the regular detecting
                    can use the TransferLearning for changing the model to work on a diffrent data set
