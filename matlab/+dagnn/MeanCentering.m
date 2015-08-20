classdef MeanCentering < dagnn.ElementWise 
  properties
    mean = 0;
  end
  
  methods
    function outputs = forward(obj, inputs, ~)
      outputs = cell(numel(inputs),1) ;
      
      for i=1:numel(inputs)
        outputs{i} = bsxfun(@minus, inputs{i}, obj.mean) ;
      end
    end
    
    function [derInputs, derParams] = backward(~, ~, ~, derOutputs)
      derInputs = derOutputs ;
      derParams = {} ;
    end
    
    function obj = MeanCentering(varargin)
      obj.load(varargin) ;
    end
  end
  
end

