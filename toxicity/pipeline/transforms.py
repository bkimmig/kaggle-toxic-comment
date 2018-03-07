import pandas as pd
import numpy as np


class DataTransformer(object):
    
    def __init__(
            self,
            transforms,
    ):
        self.transforms = transforms
    
    def __call__(
            self,
            data_ids,
            controls=None,
    ):
        if controls is None:
            controls = {}
        data = {
            "id":data_ids,
        }
        data.update(controls)
        
        for transform in self.transforms:
            data = transform(data)
        return data


class DataFormatter(object):

    def __init__(
            self,
            format_dict,
    ):
        self.format_dict = format_dict
    
    def __call__(self, data_dict):
        output = {}
        for key in self.format_dict.keys():
            extractor = self.format_dict[key]
            if extractor is None:
                output[key] = data_dict[key]
            else:
                output[key] = extractor(data_dict)
        return output

        
class DataFrameTransformWrapper(object):
    
    def __init__(
        self, 
        data_frame,
        target_column,
        output_key,
        as_array=True,
    ):
        self.data_frame = data_frame
        self.target_column = target_column
        self.output_key = output_key
        self.as_array = as_array
    
    def __call__(self, data_in):
        ids = data_in["id"]
        targ = self.data_frame[self.target_column].loc[ids]
        if self.as_array:
            targ = targ.values
        
        #put the extracted data back into the data dictionary and return the whole dictionary
        data_in[self.output_key] = targ
        
        return data_in

class FunctionTransformWrapper(object):
    
    def __init__(
        self, 
        transform_function,
        input_key,
        output_key,
    ):
        self.transform_function = transform_function
        self.input_key = input_key
        self.output_key = output_key
    
    def __call__(self, data_in):
        targ = self.transform_function(data_in[self.input_key])
        data_in[self.output_key] = targ
        return data_in




